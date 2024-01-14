package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
)

type Options struct {
	Runner

	// Predict options used at runtime
	NumKeep          int      `json:"num_keep,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	TFSZ             float32  `json:"tfs_z,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	Mirostat         int      `json:"mirostat,omitempty"`
	MirostatTau      float32  `json:"mirostat_tau,omitempty"`
	MirostatEta      float32  `json:"mirostat_eta,omitempty"`
	PenalizeNewline  bool     `json:"penalize_newline,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// Runner options which must be set when the model is loaded into memory
type Runner struct {
	UseNUMA            bool    `json:"numa,omitempty"`
	NumCtx             int     `json:"num_ctx,omitempty"`
	NumBatch           int     `json:"num_batch,omitempty"`
	NumGQA             int     `json:"num_gqa,omitempty"`
	NumGPU             int     `json:"num_gpu,omitempty"`
	MainGPU            int     `json:"main_gpu,omitempty"`
	LowVRAM            bool    `json:"low_vram,omitempty"`
	F16KV              bool    `json:"f16_kv,omitempty"`
	LogitsAll          bool    `json:"logits_all,omitempty"`
	VocabOnly          bool    `json:"vocab_only,omitempty"`
	UseMMap            bool    `json:"use_mmap,omitempty"`
	UseMLock           bool    `json:"use_mlock,omitempty"`
	EmbeddingOnly      bool    `json:"embedding_only,omitempty"`
	RopeFrequencyBase  float32 `json:"rope_frequency_base,omitempty"`
	RopeFrequencyScale float32 `json:"rope_frequency_scale,omitempty"`
	NumThread          int     `json:"num_thread,omitempty"`
}

type ImageData []byte

type GenerateRequest struct {
	Model    string      `json:"model"`
	Prompt   string      `json:"prompt"`
	System   string      `json:"system"`
	Template string      `json:"template"`
	Context  []int       `json:"context,omitempty"`
	Stream   *bool       `json:"stream,omitempty"`
	Raw      bool        `json:"raw,omitempty"`
	Format   string      `json:"format"`
	Images   []ImageData `json:"images,omitempty"`

	Options map[string]interface{} `json:"options"`
}

type Data struct { // https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
	Model   string  `json:"model"`
	Prompt  string  `json:"prompt"`
	System  string  `json:"system"`
	Options Options `json:"options"`
}

type Client struct {
	base *url.URL
	http http.Client
}

const maxBufferSize = 65535 // int

func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict:       -1, //Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
		NumKeep:          0,
		Temperature:      1.0, //The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
		TopK:             40,  //Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
		TopP:             0.9, //Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
		TFSZ:             1.0, //Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
		TypicalP:         1.0,
		RepeatLastN:      64,  //Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
		RepeatPenalty:    1.1, //Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Mirostat:         0,   //Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
		MirostatTau:      5.0, //Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)
		MirostatEta:      0.1, //Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)
		PenalizeNewline:  true,
		Seed:             -1, //Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)

		Runner: Runner{
			// options set when the model is loaded
			NumCtx:             4096, //Sets the size of the context window used to generate the next token. (Default: 2048)
			RopeFrequencyBase:  10000.0,
			RopeFrequencyScale: 1.0,
			NumBatch:           512,
			NumGPU:             -1, //The number of layers to send to the GPU(s). On macOS it defaults to 1 to enable metal support, 0 to disable. // -1 here indicates that NumGPU should be set dynamically
			NumGQA:             1,
			NumThread:          15, //0, // let the runtime decide
			LowVRAM:            false,
			F16KV:              true,
			UseMLock:           false,
			UseMMap:            true,
			UseNUMA:            false,
			EmbeddingOnly:      true,
		},
	}
}

/*func structToMap(obj interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	val := reflect.ValueOf(obj)

	for i := 0; i < val.Type().NumField(); i++ {
		field := val.Type().Field(i)
		result[field.Name] = val.Field(i).Interface()
	}
	fmt.Println(result)
	return result
}*/

func (c *Client) stream(method string, data []byte) error { //fn func([]byte) error) error {
	buf := bytes.NewBuffer(data)

	requestURL := "HTTP://" + c.base.Host + c.base.Path
	request, err := http.NewRequest(method, requestURL, buf)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/x-ndjson")
	request.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", "0.1.20", runtime.GOARCH, runtime.GOOS, runtime.Version()))

	response, err := c.http.Do(request)
	if err != nil {
		return err
	}

	defer response.Body.Close()

	scanner := bufio.NewScanner(response.Body)
	// increase the buffer size to avoid running out of space
	scanBuf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(scanBuf, maxBufferSize)
	var result map[string]interface{}
	for scanner.Scan() {
		var errorResponse struct {
			Error string `json:"error,omitempty"`
		}

		bts := scanner.Bytes()
		if err := json.Unmarshal(bts, &errorResponse); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if errorResponse.Error != "" {
			return fmt.Errorf(errorResponse.Error)
		}

		if response.StatusCode >= http.StatusBadRequest {
			os.Exit(1)
			//return StatusError{
			//	StatusCode:   response.StatusCode,
			//	Status:       response.Status,
			//	ErrorMessage: errorResponse.Error,
			//}
		}

		//if err := fn(bts); err != nil {
		//	return err
		//}

		json.Unmarshal(bts, &result)
		fmt.Printf("%s", result["response"])
	}
	fmt.Print("\n")
	return nil
}

func main() {
	var c Client
	c.base = &url.URL{
		Host: "127.0.0.1:6666",
		Path: "/api/generate",
	}

	//url := "http://127.0.0.1:6666/api/generate"

	var message string
	var err error
	for {
		fmt.Print("YOU: ")
		reader := bufio.NewReader(os.Stdin)
		message, err = reader.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading input:", err)
			os.Exit(1)
		}

		if strings.Compare(message, "exit\n") == 0 {
			os.Exit(0)
		}

		data := Data{
			Model:  "AIMY3",
			Prompt: message,
			//System:  "You are AIMY, a dedicated and helpfull AI that authors latex language books as extensively as possible while exploring knowledge as vast as it may be to complete the reader's understanding of the matter of the books.",
			Options: DefaultOptions(),
		}

		//fmt.Println(data)

		dataM, err := json.Marshal(data)
		if err != nil {
			panic(err)
		}

		var buffer []byte //512 Kilobytes

		fmt.Print("AIMY: ")
		for {
			err = c.stream(http.MethodPost, dataM)
			if err != nil {
				os.Exit(1)
			}
			if len(buffer) == 0 {
				break
			}
		}
	}

}
