package llm

import (
	"path/filepath"
	"testing"

	"github.com/sigpanic/goink/internal/llm/adapter"
	"github.com/stretchr/testify/require"
)

func TestCodexCompatibilitySurvivesEncryptedConfigurationRoundTrip(t *testing.T) {
	view := &LLMConfigView{Providers: []ProviderView{{
		AdapterName: adapter.ProtocolOpenAIResponses, ProviderType: "openai-codex", Name: "Custom Responses",
		Source: "custom", ChatURL: "https://relay.example/v1/responses", APIKey: "test-key",
		CustomModels: []ModelInfo{{ID: "gpt-6.1-sol", Name: "GPT", ContextWindow: 200000}},
	}}}
	config, err := view.ToUserConfig()
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "llm_config.enc")
	require.NoError(t, SaveUserConfig(path, config))
	restored, err := LoadUserConfig(path)
	require.NoError(t, err)
	merged, err := Merge(restored)
	require.NoError(t, err)
	require.Equal(t, "openai-codex", merged[adapter.ProtocolOpenAIResponses]["Custom Responses"].ProviderType)
	result, err := BuildConfigView(restored)
	require.NoError(t, err)
	for _, provider := range result.Providers {
		if provider.Name == "Custom Responses" {
			require.Equal(t, "openai-codex", provider.ProviderType)
			require.Equal(t, view.Providers[0].CustomModels, provider.CustomModels)
			return
		}
	}
	t.Fatal("custom Responses provider disappeared after configuration reload")
}
