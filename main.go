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

// --- Configuration Structures and Logic ---

// Config holds the application settings loaded from config.json.
type Config struct {
	OllamaURL   string `json:"ollama_url"`
	OllamaModel string `json:"ollama_model"`
}

// Default configuration settings
const defaultOllamaURL = "http://localhost:11434/api/chat"
const defaultOllamaModel = "llama3"

// Global variable to hold the loaded configuration
var cfg Config

// getConfigFilePath determines the cross-platform path for the config file.
func getConfigFilePath() (string, error) {
	var dir string
	const appName = "shai"

	switch runtime.GOOS {
	case "windows":
		dir = os.Getenv("APPDATA")
	case "darwin": // macOS
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, "Library", "Application Support")
	case "linux": // Linux and other Unix-like systems
		dir = os.Getenv("XDG_CONFIG_HOME")
		if dir == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			dir = filepath.Join(home, ".config")
		}
	default:
		// Fallback for unknown systems
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(home, "."+appName)
	}

	// Ensure the application specific directory exists
	appDir := filepath.Join(dir, appName)
	if err := os.MkdirAll(appDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create config directory %s: %w", appDir, err)
	}

	return filepath.Join(appDir, "config.json"), nil
}

// loadConfig reads the configuration file or creates a default one.
func loadConfig() error {
	configPath, err := getConfigFilePath()
	if err != nil {
		return fmt.Errorf("failed to get config path: %w", err)
	}

	// Check if config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// File does not exist, create default config
		cfg = Config{
			OllamaURL:   defaultOllamaURL,
			OllamaModel: defaultOllamaModel,
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

	// File exists, load config
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file %s: %w", configPath, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("failed to parse config file %s: %w", configPath, err)
	}

	return nil
}

// --- Agent Logic Constants and System Prompt ---

// systemPromptTemplate is a template to be formatted with the user's task, OS, and shell.
const systemPromptTemplate = `You are an autonomous shell agent called 'shai' (Shell AI).

YOUR CORE MISSION: %s

CURRENT ENVIRONMENT:
OS: %s
SHELL: %s
PWD: %s

RULES:
1. I will send you the result of the previous command or user input as a 'user' message.
2. After executing a command that *should* complete the task, you MUST execute a final verification command (e.g., 'ls', 'cat', 'grep') and confirm the output matches the goal before proceeding.
3. You MUST strictly adhere to the following output protocol, starting with the action keyword:
   - To run a command: Use "RUN" followed by the command on the same line or the next line. The command MUST NOT contain any code fences.
   - To ask for clarification: Use "ASK" followed by the question on the same line or the next line.
   - If the task is VERIFIED and the goal state is achieved, output ONLY "TASK_COMPLETE".
   - If you determine the task cannot be completed or requires external human action, output ONLY "TASK_STOPPED".
4. Your command lines MUST be a single line appropriate for the detected SHELL.

Your first response, when you receive "START", MUST be the first action (RUN or ASK).
`

// --- Structures for Ollama API Interaction ---

// Message structure for the Ollama /api/chat endpoint
type Message struct {
	Role    string `json:"role"` // 'user', 'assistant', or 'system'
	Content string `json:"content"`
}

// Request structure for the Ollama /api/chat endpoint
type ChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	KeepAlive string    `json:"keep_alive"`
}

// Response structure for the Ollama /api/chat endpoint (non-streaming)
type ChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   Message   `json:"message"` // The assistant's response message
	Done      bool      `json:"done"`
}

// --- Core Agent Logic ---

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: shai \"<task description>\"")
		fmt.Println("Example: shai \"convert all files under this dir from flac to mp3\"")
		os.Exit(1)
	}

	// Load configuration first
	if err := loadConfig(); err != nil {
		log.Fatalf("Fatal Error loading configuration: %v", err)
	}

	// 1. Get environment details for the system prompt
	userShell := os.Getenv("SHELL")
	// Clean up shell path on Windows or set sensible defaults
	if runtime.GOOS == "windows" {
		if strings.Contains(strings.ToLower(userShell), "powershell") {
			userShell = "powershell.exe"
		} else {
			userShell = "cmd.exe"
		}
	} else if userShell == "" {
		// Default to bash if $SHELL is not set on Unix
		userShell = "/bin/bash"
	}
	currentOS := runtime.GOOS

	// 2. Combine all command line arguments into the initial task
	initialTask := strings.Join(os.Args[1:], " ")

	// 3. Create the dynamic system instruction
	fullSystemPrompt := fmt.Sprintf(systemPromptTemplate, initialTask, currentOS, userShell, getwd())

	fmt.Printf("👋 shai initialized with task: %s\n", initialTask)
	fmt.Printf("Platform: %s | Shell: %s\n", currentOS, userShell)
	// CHANGED: Report loaded config values
	fmt.Printf("Using Ollama URL: %s | Model: %s\n", cfg.OllamaURL, cfg.OllamaModel)

	// 4. Run the agent, passing the detected shell for execution
	err := runAgent(fullSystemPrompt, userShell)
	if err != nil {
		log.Fatalf("Agent error: %v", err)
	}
}

// runAgent contains the main agent loop.
func runAgent(fullSystemPrompt string, userShell string) error {
	// Initialize message history. The first message is the 'user' starting the task.
	messages := []Message{
		{Role: "user", Content: "START"},
	}
	step := 1
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Printf("\n--- Step %d ---\n", step)

		// 1. Call Ollama to get the next instruction/command
		fmt.Println("🤔 shai is thinking...")
		response, err := callOllama(messages, fullSystemPrompt)
		if err != nil {
			return fmt.Errorf("Ollama API call failed: %w", err)
		}

		// Record the assistant's response (action) in history immediately
		messages = append(messages, Message{Role: "assistant", Content: response})

		// 2. Protocol Parsing: Split response into action and content (Robust/Lenient)
		modelOutput := strings.TrimSpace(response)
		action := ""
		content := ""

		// Find the index of the first space or newline
		idxSeparator := strings.IndexFunc(modelOutput, func(r rune) bool {
			return r == ' ' || r == '\n'
		})

		if idxSeparator == -1 {
			// If no space or newline, the whole output is the action (e.g., TASK_COMPLETE)
			action = strings.ToUpper(modelOutput)
			content = ""
		} else {
			// Action is the substring up to the first space/newline
			action = strings.ToUpper(modelOutput[:idxSeparator])
			// Content is the rest of the string, trimmed
			content = strings.TrimSpace(modelOutput[idxSeparator+1:])
		}

		// 3. Handle terminal states
		if action == "TASK_COMPLETE" {
			fmt.Println("✅ shai has completed the task successfully.")
			return nil
		}
		if action == "TASK_STOPPED" {
			fmt.Println("🛑 shai has stopped the task, as it cannot proceed or needs human input.")
			return nil
		}

		// 4. Handle RUN action
		if action == "RUN" {
			if content == "" {
				fmt.Printf("⚠️ shai provided a malformed RUN command (missing command line). Response:\n---\n%s\n---\n", modelOutput)
				// Feedback for the model
				messages = append(messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("CRITICAL ERROR: Previous response was RUN but provided no command. Full response was:\n%s", modelOutput),
				})
				step++
				continue
			}

			command := content
			if !confirmAction(fmt.Sprintf("shai wants to run this command:\n\n  $ %s\n\nAllow?", command), reader) {
				return fmt.Errorf("user rejected command, terminating")
			}

			// Execute the command, passing the correct shell path
			fmt.Printf("🚀 Running command via %s...\n", userShell)
			status, output, _ := executeCommand(command, userShell)

			// 5. Build the feedback prompt for the next loop iteration (as a new USER message)
			var feedback strings.Builder
			feedback.WriteString("PREVIOUS_COMMAND_RESULT:\n")
			feedback.WriteString(fmt.Sprintf("STATUS: %s\n", status))
			feedback.WriteString("OUTPUT:\n")
			feedback.WriteString(output)
			feedback.WriteString("\n\n")

			// Append the result as a new user message for the history
			messages = append(messages, Message{
				Role:    "user",
				Content: feedback.String(),
			})

			// 6. Handle ASK action
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

			// Get user input for the question
			fmt.Print("Your response to shai: ")
			userInput, _ := reader.ReadString('\n')

			// The user's response becomes the next prompt (new USER message).
			messages = append(messages, Message{
				Role:    "user",
				Content: fmt.Sprintf("USER_CLARIFICATION: %s", strings.TrimSpace(userInput)),
			})

			// 7. Handle Unrecognized action
		} else {
			fmt.Printf("⚠️ shai provided an UNRECOGNIZED response. Model response was:\n---\n%s\n---\n", modelOutput)
			if !confirmAction("shai provided an unparseable response. Continue the loop?", reader) {
				return fmt.Errorf("user rejected unparseable model output, terminating")
			}
			// Feed the entire unparseable output back to the model as an error state (new USER message)
			messages = append(messages, Message{
				Role:    "user",
				Content: fmt.Sprintf("UNPARSEABLE_RESPONSE_ERROR: Your previous response did not follow the protocol. Your previous output was:\n%s", modelOutput),
			})
		}

		step++
	}
}

// callOllama sends the message history and the system instruction to the Ollama Chat API and returns the assistant's message content.
func callOllama(messages []Message, systemInstruction string) (string, error) {
	// Prepend the system instruction as the first message
	fullMessages := []Message{
		{Role: "system", Content: systemInstruction},
	}
	fullMessages = append(fullMessages, messages...)

	reqBody := ChatRequest{
		// CHANGED: Use configuration values
		Model:     cfg.OllamaModel,
		Messages:  fullMessages,
		Stream:    false,
		KeepAlive: "5m",
	}

	jsonBody, _ := json.Marshal(reqBody)

	// CHANGED: Use configuration URL
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

// confirmAction prompts the user and returns true if they enter 'y' or 'Y'.
func confirmAction(message string, reader *bufio.Reader) bool {
	// The user's input determines if the action is confirmed.
	fmt.Printf("\n%s [Y/n]: ", message)

	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(strings.ToLower(input))

	// Returns true for 'y' or empty input (default), false for 'n'.
	return input != "n"
}

// executeCommand runs a shell command using the specified shell path.
func executeCommand(command string, shellPath string) (status string, output string, err error) {
	var cmd *exec.Cmd

	// On Unix-like systems, use the shell path with the -c flag.
	if runtime.GOOS != "windows" {
		cmd = exec.Command(shellPath, "-c", command)
	} else if strings.EqualFold(shellPath, "powershell.exe") || strings.EqualFold(shellPath, "powershell") {
		// Use -Command for powershell
		cmd = exec.Command("powershell.exe", "-Command", command)
	} else {
		// Default to cmd /C for other Windows cases
		cmd = exec.Command("cmd.exe", "/C", command)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Start the command
	execErr := cmd.Run()

	// Check the execution error
	if execErr != nil {
		status = "ERROR"
		// Send both the error object's string and the stderr content back to the model
		output = fmt.Sprintf("Command failed with error: %v\n%s", execErr, stderr.String())
		return status, output, nil // Return nil error to continue agent loop
	}

	status = "SUCCESS"
	// Combine stdout and stderr for the model's benefit
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
