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

package org.springframework.ai.openai.chat;

import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.OpenAiTestConfiguration;
import org.springframework.ai.retry.NonTransientAiException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.util.StringUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Live integration tests for the OpenAI raw-passthrough send path
 * ({@code OpenAiChatModel#callRaw} / {@code streamRaw}), driving spring-ai the same way
 * ai-router does: a raw, pre-serialized JSON body forwarded verbatim, with the model
 * supplied via the typed prompt options.
 *
 * <p>
 * Covers the areas requested in the PR review: token counts, tool calls, provider errors,
 * and a reasoning model. Requires {@code OPENAI_API_KEY}.
 */
@SpringBootTest(classes = OpenAiTestConfiguration.class)
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
class OpenAiChatModelRawPassthroughIT {

	private static final String DEFAULT_MODEL = "gpt-4o-mini";

	@Autowired
	private OpenAiChatModel openAiChatModel;

	private Prompt promptWithModel(String model, String userText) {
		return new Prompt(List.of(new UserMessage(userText)), OpenAiChatOptions.builder().model(model).build());
	}

	@Test
	void callRawReturnsContentAndTokenCounts() {
		String rawBody = """
				{"messages":[{"role":"user","content":"Reply with exactly: pong"}]}
				""";

		ChatResponse response = this.openAiChatModel.callRaw(promptWithModel(DEFAULT_MODEL, "Reply with exactly: pong"),
				rawBody);

		assertThat(response).isNotNull();
		assertThat(response.getResult().getOutput().getText()).isNotBlank();
		Usage usage = response.getMetadata().getUsage();
		assertThat(usage).isNotNull();
		assertThat(usage.getPromptTokens()).isGreaterThan(0);
		assertThat(usage.getCompletionTokens()).isGreaterThan(0);
		assertThat(usage.getTotalTokens()).isGreaterThan(0);
	}

	@Test
	void streamRawAccumulatesTokenCounts() {
		String rawBody = """
				{"messages":[{"role":"user","content":"Count from 1 to 5."}]}
				""";

		List<ChatResponse> responses = this.openAiChatModel
			.streamRaw(promptWithModel(DEFAULT_MODEL, "Count from 1 to 5."), rawBody)
			.collectList()
			.block();

		assertThat(responses).isNotNull().isNotEmpty();

		String content = responses.stream()
			.map(r -> r.getResult() != null ? r.getResult().getOutput().getText() : "")
			.collect(Collectors.joining());
		assertThat(content).isNotBlank();

		// The final usage chunk must be accumulated even though include_usage was forced
		// onto the raw wire body, not the typed request (the streaming-usage fix).
		boolean anyUsage = responses.stream()
			.map(r -> r.getMetadata().getUsage())
			.anyMatch(u -> u != null && u.getTotalTokens() != null && u.getTotalTokens() > 0);
		assertThat(anyUsage).isTrue();
	}

	@Test
	void callRawSurfacesToolCallsWithoutExecutingThem() {
		String rawBody = """
				{
				  "messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
				  "tools":[{"type":"function","function":{
				    "name":"get_weather",
				    "description":"Get the current weather for a city",
				    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
				  }}],
				  "tool_choice":"required"
				}
				""";

		// No tool callbacks + internal tool execution disabled: the raw path must forward
		// the tool call to the caller (like ai-router), not execute it locally.
		Prompt prompt = new Prompt(List.of(new UserMessage("What is the weather in Paris? Use the tool.")),
				OpenAiChatOptions.builder().model(DEFAULT_MODEL).internalToolExecutionEnabled(false).build());

		ChatResponse response = this.openAiChatModel.callRaw(prompt, rawBody);

		assertThat(response).isNotNull();
		assertThat(response.hasToolCalls()).isTrue();
		assertThat(response.getResult().getOutput().getToolCalls().get(0).name()).isEqualTo("get_weather");
	}

	@Test
	void callRawPropagatesProviderError() {
		// temperature 5 is out of OpenAI's accepted [0,2] range -> 400 from the provider.
		String rawBody = """
				{"messages":[{"role":"user","content":"hi"}],"temperature":5}
				""";

		assertThatThrownBy(() -> this.openAiChatModel.callRaw(promptWithModel(DEFAULT_MODEL, "hi"), rawBody))
			.isInstanceOf(NonTransientAiException.class);
	}

	@Test
	void callRawWorksForReasoningModel() {
		// A reasoning model has different parameter rules (no temperature, uses
		// max_completion_tokens); this is a smoke test that the raw path forwards such a
		// request without choking and still reports usage.
		String rawBody = """
				{"messages":[{"role":"user","content":"What is 17 + 25? Reply with just the number."}]}
				""";

		ChatResponse response = this.openAiChatModel.callRaw(promptWithModel("o4-mini", "What is 17 + 25?"), rawBody);

		assertThat(response).isNotNull();
		assertThat(StringUtils.hasText(response.getResult().getOutput().getText())).isTrue();
		assertThat(response.getMetadata().getUsage().getTotalTokens()).isGreaterThan(0);
	}

}
