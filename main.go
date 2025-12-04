package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type Config struct {
	OllamaURL   string `json:"ollama_url"`
	OllamaModel string `json:"ollama_model"`
	AdditionalContext string `json:"additional_context"`
}

const defaultOllamaURL = "http://localhost:11434/api/chat"
const defaultOllamaModel = "llama3"
const defaultAdditionalContext = ""

var cfg Config

func getConfigFilePath() (string, error) {
	var dir string
	const appName = "shai"

	switch runtime.GOOS {
	case "windows":
		dir = os.Getenv("APPDATA")
	case "darwin":
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, ".config")
	case "linux":
		dir = os.Getenv("XDG_CONFIG_HOME")
		if dir == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			dir = filepath.Join(home, ".config")
		}
	default:
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, "."+appName)
	}

	appDir := filepath.Join(dir, appName)
	if err := os.MkdirAll(appDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create config directory %s: %w", appDir, err)
	}

	return filepath.Join(appDir, "config.json"), nil
}

func loadConfig() error {
	configPath, err := getConfigFilePath()
	if err != nil {
		return fmt.Errorf("failed to get config path: %w", err)
	}

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		cfg = Config{
			OllamaURL:   defaultOllamaURL,
			OllamaModel: defaultOllamaModel,
			AdditionalContext: defaultAdditionalContext,
		}
		fmt.Printf("⚠️ Configuration file not found. Creating default config at: %s\n", configPath)

		data, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal default config: %w", err)
		}

		if err := os.WriteFile(configPath, data, 0644); err != nil {
			return fmt.Errorf("failed to write default config: %w", err)
		}

		return nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file %s: %w", configPath, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("failed to parse config file %s: %w", configPath, err)
	}

	return nil
}

const systemPromptTemplate = `You are an autonomous shell agent called 'shai' (Shell AI).

CURRENT USER REQUEST: %s

CURRENT ENVIRONMENT:
Operating System: %s
Shell: %s
Current Working Directory: %s

RULES:
1. I will send you the result of the previous command or user input as a 'user' message.
2. After executing a command that *should* complete the task, you MUST execute a final verification command (e.g., 'ls', 'cat', 'grep') and confirm the output matches the goal before proceeding.
3. You MUST strictly adhere to the following output protocol, starting with the action keyword:
   - To run a command: Use "RUN" followed by the command on the same line or the next line. The command MUST NOT contain any code fences, explanation, or commentary of any kind.
   - To ask for clarification: Use "ASK" followed by the question on the same line or the next line.
   - If the task is VERIFIED and the goal state is achieved, output "TASK_COMPLETE" followed by any additional information.
   - If you determine the task cannot be completed or requires external human action, output "TASK_STOPPED" followed by any additional information.
4. Your command lines MUST be a single line appropriate current environment's shell.
5. Assume that your commands are being run in the current working directory.
6. Do not ask questions which you could find the answer to yourself by running commands (such as "is X package installed?"). Find the answer for yourself whenever possible.
7. Do not ask questions you already know the answer to.
%s
Your first response, when you receive "START", MUST be the first action (RUN or ASK).
`

const additionalContextTemplate = `
ADDITIONAL CONTEXT:
%s
`

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	KeepAlive string    `json:"keep_alive"`
}

type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"`
	Done      bool      `json:"done"`
}


func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: shai \"<task description>\"")
		fmt.Println("Example: shai \"convert all files under this dir from flac to mp3\"")
		os.Exit(1)
	}

	if err := loadConfig(); err != nil {
		log.Fatalf("Fatal Error loading configuration: %v", err)
	}

	userShell := os.Getenv("SHELL")
	if runtime.GOOS == "windows" {
		if strings.Contains(strings.ToLower(userShell), "powershell") {
			userShell = "powershell.exe"
		} else {
			userShell = "cmd.exe"
		}
	} else if userShell == "" {
		userShell = "/bin/bash"
	}
	currentOS := runtime.GOOS

	initialTask := strings.Join(os.Args[1:], " ")

	fullSystemPrompt := generateSystemPrompt(initialTask, currentOS, userShell)

	fmt.Printf("👋 shai initialized with task: %s\n", initialTask)
	fmt.Printf("Platform: %s | Shell: %s\n", currentOS, userShell)
	fmt.Printf("Using Ollama URL: %s | Model: %s\n", cfg.OllamaURL, cfg.OllamaModel)

	err := runAgent(fullSystemPrompt, userShell)
	if err != nil {
		log.Fatalf("Agent error: %v", err)
	}
}

func runAgent(fullSystemPrompt string, userShell string) error {
	messages := []Message{
		{Role: "user", Content: "START"},
	}
	step := 1
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Printf("\n--- Step %d ---\n", step)

		fmt.Println("🤔 shai is thinking...")
		response, err := callOllama(messages, fullSystemPrompt)
		if err != nil {
			return fmt.Errorf("Ollama API call failed: %w", err)
		}

		messages = append(messages, Message{Role: "assistant", Content: response})

		modelOutput := strings.TrimSpace(response)
		action := ""
		content := ""

		idxSeparator := strings.IndexFunc(modelOutput, func(r rune) bool {
			return r == ' ' || r == '\n'
		})

		if idxSeparator == -1 {
			action = strings.ToUpper(modelOutput)
			content = ""
		} else {
			action = strings.ToUpper(modelOutput[:idxSeparator])
			content = strings.TrimSpace(modelOutput[idxSeparator+1:])
		}

		if action == "TASK_COMPLETE" {
			fmt.Println("✅ shai has completed the task successfully.")
			fmt.Println(content)
			return nil
		}
		if action == "TASK_STOPPED" {
			fmt.Println("🛑 shai has stopped the task, as it cannot proceed or needs human input.")
			fmt.Println(content)
			return nil
		}

		if action == "RUN" {
			if content == "" {
				fmt.Printf("⚠️ shai provided a malformed RUN command (missing command line). Response:\n---\n%s\n---\n", modelOutput)
				messages = append(messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("CRITICAL ERROR: Previous response was RUN but provided no command. Full response was:\n%s", modelOutput),
				})
				step++
				continue
			}

			command := content
			if !confirmAction(fmt.Sprintf("✨ shai wants to run this command:\n\n  $ %s\n\nAllow?", command), reader) {
				return fmt.Errorf("user rejected command, terminating")
			}

			fmt.Printf("🚀 Running command via %s...\n", userShell)
			status, output, _ := executeCommand(command, userShell)

			var feedback strings.Builder
			feedback.WriteString("PREVIOUS_COMMAND_RESULT:\n")
			feedback.WriteString(fmt.Sprintf("STATUS: %s\n", status))
			feedback.WriteString("OUTPUT:\n")
			feedback.WriteString(output)
			feedback.WriteString("\n\n")

			messages = append(messages, Message{
				Role:    "user",
				Content: feedback.String(),
			})

		} else if action == "ASK" {
			if content == "" {
				fmt.Printf("⚠️ shai provided a malformed ASK request (missing question). Response:\n---\n%s\n---\n", modelOutput)
				// Feedback for the model
				messages = append(messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("CRITICAL ERROR: Previous response was ASK but provided no question. Full response was:\n%s", modelOutput),
				})
				step++
				continue
			}

			question := content
			fmt.Printf("\n❓ shai needs clarification:\n%s\n", question)

			fmt.Print("Your response to shai: ")
			userInput, _ := reader.ReadString('\n')

			messages = append(messages, Message{
				Role:    "user",
				Content: fmt.Sprintf("USER_CLARIFICATION: %s", strings.TrimSpace(userInput)),
			})

		} else {
			fmt.Printf("⚠️ shai provided an UNRECOGNIZED response. Model response was:\n---\n%s\n---\n", modelOutput)
			if !confirmAction("shai provided an unparseable response. Continue the loop?", reader) {
				return fmt.Errorf("user rejected unparseable model output, terminating")
			}
			messages = append(messages, Message{
				Role:    "user",
				Content: fmt.Sprintf("UNPARSEABLE_RESPONSE_ERROR: Your previous response did not follow the protocol. Your previous output was:\n%s", modelOutput),
			})
		}

		step++
	}
}

func callOllama(messages []Message, systemInstruction string) (string, error) {
	fullMessages := []Message{
		{Role: "system", Content: systemInstruction},
	}
	fullMessages = append(fullMessages, messages...)

	reqBody := ChatRequest{
		Model:     cfg.OllamaModel,
		Messages:  fullMessages,
		Stream:    false,
		KeepAlive: "5m",
	}

	jsonBody, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", cfg.OllamaURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request to Ollama: %w. Is Ollama running at %s?", err, cfg.OllamaURL)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Ollama API returned non-200 status code: %d. Body: %s", resp.StatusCode, string(bodyBytes))
	}

	var ollamaResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return "", fmt.Errorf("failed to decode Ollama chat response: %w", err)
	}

	return ollamaResp.Message.Content, nil
}

func confirmAction(message string, reader *bufio.Reader) bool {
	fmt.Printf("\n%s [Y/n/q]: ", message)

	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(strings.ToLower(input))

	if input == "q" {
		os.Exit(0)
	}

	return input != "n"
}

func executeCommand(command string, shellPath string) (status string, output string, err error) {
	var cmd *exec.Cmd

	if runtime.GOOS != "windows" {
		cmd = exec.Command(shellPath, "-c", command)
	} else if strings.EqualFold(shellPath, "powershell.exe") || strings.EqualFold(shellPath, "powershell") {
		cmd = exec.Command("powershell.exe", "-Command", command)
	} else {
		cmd = exec.Command("cmd.exe", "/C", command)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	execErr := cmd.Run()

	if execErr != nil {
		status = "ERROR"
		output = fmt.Sprintf("Command failed with error: %v\n%s", execErr, stderr.String())
		return status, output, nil
	}

	status = "SUCCESS"
	output = fmt.Sprintf("STDOUT:\n%s\nSTDERR:\n%s", stdout.String(), stderr.String())

	return status, output, nil
}

func getwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "UNKONWN"
	}
	return wd
}

func generateSystemPrompt(initialTask string, currentOS string, userShell string) string {
	if cfg.AdditionalContext == "" {
		return fmt.Sprintf(systemPromptTemplate, initialTask, currentOS, userShell, getwd(), "")
	} else {
		return fmt.Sprintf(systemPromptTemplate, initialTask, currentOS, userShell, getwd(), fmt.Sprintf(additionalContextTemplate, cfg.AdditionalContext))	
	}
}
