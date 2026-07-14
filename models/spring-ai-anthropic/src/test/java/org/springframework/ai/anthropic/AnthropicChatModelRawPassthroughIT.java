/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.anthropic;

import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Live integration tests for the Anthropic raw-passthrough send path
 * ({@code AnthropicChatModel#callRaw} / {@code streamRaw}), driving spring-ai the same
 * way ai-router does: a raw, pre-serialized JSON body forwarded verbatim, with the model
 * supplied via the typed prompt options.
 *
 * <p>
 * Covers token counts, tool calls and provider errors on the raw path. Requires
 * {@code ANTHROPIC_API_KEY}.
 */
@SpringBootTest(classes = AnthropicTestConfiguration.class)
@EnabledIfEnvironmentVariable(named = "ANTHROPIC_API_KEY", matches = ".+")
class AnthropicChatModelRawPassthroughIT {

	private static final String DEFAULT_MODEL = "claude-3-5-haiku-latest";

	@Autowired
	private AnthropicChatModel anthropicChatModel;

	private Prompt promptWithModel(String model, String userText) {
		return new Prompt(List.of(new UserMessage(userText)), AnthropicChatOptions.builder().model(model).build());
	}

	@Test
	void callRawReturnsContentAndTokenCounts() {
		String rawBody = """
				{"messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":64}
				""";

		ChatResponse response = this.anthropicChatModel
			.callRaw(promptWithModel(DEFAULT_MODEL, "Reply with exactly: pong"), rawBody);

		assertThat(response).isNotNull();
		assertThat(response.getResult().getOutput().getText()).isNotBlank();
		Usage usage = response.getMetadata().getUsage();
		assertThat(usage).isNotNull();
		assertThat(usage.getPromptTokens()).isGreaterThan(0);
		assertThat(usage.getCompletionTokens()).isGreaterThan(0);
	}

	@Test
	void callRawInjectsMaxTokensWhenAbsent() {
		// Anthropic rejects requests without max_tokens; the raw path must inject the
		// gateway default when the client body omits it.
		String rawBody = """
				{"messages":[{"role":"user","content":"Reply with exactly: pong"}]}
				""";

		ChatResponse response = this.anthropicChatModel
			.callRaw(promptWithModel(DEFAULT_MODEL, "Reply with exactly: pong"), rawBody);

		assertThat(response).isNotNull();
		assertThat(response.getResult().getOutput().getText()).isNotBlank();
	}

	@Test
	void streamRawAccumulatesTokenCounts() {
		String rawBody = """
				{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":256}
				""";

		List<ChatResponse> responses = this.anthropicChatModel
			.streamRaw(promptWithModel(DEFAULT_MODEL, "Count from 1 to 5."), rawBody)
			.collectList()
			.block();

		assertThat(responses).isNotNull().isNotEmpty();

		String content = responses.stream()
			.map(r -> r.getResult() != null && r.getResult().getOutput().getText() != null
					? r.getResult().getOutput().getText() : "")
			.collect(Collectors.joining());
		assertThat(content).isNotBlank();

		// Usage arrives via message_start (input) and message_delta (output) and must be
		// accumulated on the raw path just like on the typed one.
		boolean anyUsage = responses.stream()
			.map(r -> r.getMetadata().getUsage())
			.anyMatch(u -> u != null && u.getPromptTokens() != null && u.getPromptTokens() > 0
					&& u.getCompletionTokens() != null && u.getCompletionTokens() > 0);
		assertThat(anyUsage).isTrue();
	}

	@Test
	void callRawSurfacesToolCallsWithoutExecutingThem() {
		String rawBody = """
				{
				  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
				  "max_tokens":512,
				  "tools":[{
				    "name":"get_weather",
				    "description":"Get the current weather for a city",
				    "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
				  }],
				  "tool_choice":{"type":"any"}
				}
				""";

		// No tool callbacks + internal tool execution disabled: the raw path must forward
		// the tool call to the caller (like ai-router), not execute it locally.
		Prompt prompt = new Prompt(List.of(new UserMessage("What is the weather in Paris? Use the tool.")),
				AnthropicChatOptions.builder().model(DEFAULT_MODEL).internalToolExecutionEnabled(false).build());

		ChatResponse response = this.anthropicChatModel.callRaw(prompt, rawBody);

		assertThat(response).isNotNull();
		assertThat(response.hasToolCalls()).isTrue();
		assertThat(response.getResult().getOutput().getToolCalls().get(0).name()).isEqualTo("get_weather");
	}

	@Test
	void callRawPropagatesProviderError() {
		// temperature 5 is out of Anthropic's accepted [0,1] range -> 400 from the
		// provider.
		String rawBody = """
				{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"temperature":5}
				""";

		assertThatThrownBy(() -> this.anthropicChatModel.callRaw(promptWithModel(DEFAULT_MODEL, "hi"), rawBody))
			.isInstanceOf(RuntimeException.class);
	}

}
