/*
 * Copyright 2023-present the original author or authors.
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

import org.junit.jupiter.api.Test;

import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.anthropic.api.AnthropicCacheOptions;
import org.springframework.ai.anthropic.api.AnthropicCacheStrategy;
import org.springframework.ai.anthropic.api.AnthropicCacheTtl;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Christian Tzolov
 * @author Alexandros Pappas
 * @author Thomas Vitale
 */
public class ChatCompletionRequestTests {

	@Test
	public void createRequestWithChatOptions() {

		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("DEFAULT_MODEL").temperature(66.6).build())
			.build();

		var prompt = client.buildRequestPrompt(new Prompt("Test message content"));

		var request = client.createRequest(prompt, false);

		assertThat(request.messages()).hasSize(1);
		assertThat(request.stream()).isFalse();

		assertThat(request.model()).isEqualTo("DEFAULT_MODEL");
		assertThat(request.temperature()).isEqualTo(66.6);

		prompt = client.buildRequestPrompt(new Prompt("Test message content",
				AnthropicChatOptions.builder().model("PROMPT_MODEL").temperature(99.9).build()));

		request = client.createRequest(prompt, true);

		assertThat(request.messages()).hasSize(1);
		assertThat(request.stream()).isTrue();

		assertThat(request.model()).isEqualTo("PROMPT_MODEL");
		assertThat(request.temperature()).isEqualTo(99.9);
	}

	@Test
	public void createRequestWithToolChoice() {

		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("DEFAULT_MODEL").build())
			.build();

		// Test with ToolChoiceAuto
		var autoToolChoice = new AnthropicApi.ToolChoiceAuto();
		var prompt = client.buildRequestPrompt(
				new Prompt("Test message content", AnthropicChatOptions.builder().toolChoice(autoToolChoice).build()));

		var request = client.createRequest(prompt, false);

		assertThat(request.toolChoice()).isNotNull();
		assertThat(request.toolChoice()).isInstanceOf(AnthropicApi.ToolChoiceAuto.class);
		assertThat(request.toolChoice().type()).isEqualTo("auto");

		// Test with ToolChoiceAny
		var anyToolChoice = new AnthropicApi.ToolChoiceAny();
		prompt = client.buildRequestPrompt(
				new Prompt("Test message content", AnthropicChatOptions.builder().toolChoice(anyToolChoice).build()));

		request = client.createRequest(prompt, false);

		assertThat(request.toolChoice()).isNotNull();
		assertThat(request.toolChoice()).isInstanceOf(AnthropicApi.ToolChoiceAny.class);
		assertThat(request.toolChoice().type()).isEqualTo("any");

		// Test with ToolChoiceTool
		var specificToolChoice = new AnthropicApi.ToolChoiceTool("get_weather");
		prompt = client.buildRequestPrompt(new Prompt("Test message content",
				AnthropicChatOptions.builder().toolChoice(specificToolChoice).build()));

		request = client.createRequest(prompt, false);

		assertThat(request.toolChoice()).isNotNull();
		assertThat(request.toolChoice()).isInstanceOf(AnthropicApi.ToolChoiceTool.class);
		assertThat(request.toolChoice().type()).isEqualTo("tool");
		assertThat(((AnthropicApi.ToolChoiceTool) request.toolChoice()).name()).isEqualTo("get_weather");

		// Test with ToolChoiceNone
		var noneToolChoice = new AnthropicApi.ToolChoiceNone();
		prompt = client.buildRequestPrompt(
				new Prompt("Test message content", AnthropicChatOptions.builder().toolChoice(noneToolChoice).build()));

		request = client.createRequest(prompt, false);

		assertThat(request.toolChoice()).isNotNull();
		assertThat(request.toolChoice()).isInstanceOf(AnthropicApi.ToolChoiceNone.class);
		assertThat(request.toolChoice().type()).isEqualTo("none");

		// Test with disableParallelToolUse
		var autoWithDisabledParallel = new AnthropicApi.ToolChoiceAuto(true);
		prompt = client.buildRequestPrompt(new Prompt("Test message content",
				AnthropicChatOptions.builder().toolChoice(autoWithDisabledParallel).build()));

		request = client.createRequest(prompt, false);

		assertThat(request.toolChoice()).isNotNull();
		assertThat(request.toolChoice()).isInstanceOf(AnthropicApi.ToolChoiceAuto.class);
		assertThat(((AnthropicApi.ToolChoiceAuto) request.toolChoice()).disableParallelToolUse()).isTrue();
	}

	@Test
	void agenticToolUseCachesOnlyLastToolResultMessage() {
		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("DEFAULT_MODEL").build())
			.build();

		AnthropicCacheOptions cacheOptions = AnthropicCacheOptions.builder()
			.strategy(AnthropicCacheStrategy.AGENTIC_TOOL_USE)
			.messageTypeTtl(MessageType.TOOL, AnthropicCacheTtl.FIVE_MINUTES)
			// Make the threshold huge so no delta-based tail breakpoint is chosen.
			.minCacheablePromptLength(100_000)
			.build();

		var promptOptions = AnthropicChatOptions.builder().cacheOptions(cacheOptions).build();

		List<Message> messages = List.of(new UserMessage("Original user request"), AssistantMessage.builder()
			.content("Calling tool")
			.toolCalls(
					List.of(new AssistantMessage.ToolCall("call-1", "tool_use", "get_weather", "{\"city\":\"Paris\"}")))
			.build(),
				ToolResponseMessage.builder()
					.responses(List.of(new ToolResponseMessage.ToolResponse("call-1", "get_weather", "{\"temp\":20}")))
					.build(),
				AssistantMessage.builder().content("Calling tool again").build(),
				ToolResponseMessage.builder()
					.responses(
							List.of(new ToolResponseMessage.ToolResponse("call-2", "get_time", "{\"time\":\"10:00\"}")))
					.build(),
				new UserMessage("Current question"));

		var request = client.createRequest(new Prompt(messages, promptOptions), false);

		// USER message content must not be cached (agentic skips USER for breakpoint)
		var firstUserMessage = request.messages().get(0);
		assertThat(firstUserMessage.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(firstUserMessage.content()).hasSize(1);
		assertThat(firstUserMessage.content().get(0).cacheControl()).isNull();

		// ASSISTANT tool_use blocks must not be cached for agentic
		var assistantWithToolUse = request.messages().get(1);
		assertThat(assistantWithToolUse.role()).isEqualTo(AnthropicApi.Role.ASSISTANT);
		assertThat(assistantWithToolUse.content())
			.anyMatch(cb -> cb.type() == AnthropicApi.ContentBlock.Type.TOOL_USE && cb.cacheControl() == null);

		// Only the last TOOL_RESULT message should have cache_control
		var toolResult1 = request.messages().get(2);
		assertThat(toolResult1.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(toolResult1.content()).hasSize(1);
		assertThat(toolResult1.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TOOL_RESULT);
		assertThat(toolResult1.content().get(0).cacheControl()).isNull();

		var toolResult2 = request.messages().get(4);
		assertThat(toolResult2.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(toolResult2.content()).hasSize(1);
		assertThat(toolResult2.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TOOL_RESULT);
		assertThat(toolResult2.content().get(0).cacheControl()).isNotNull();
		assertThat(toolResult2.content().get(0).cacheControl().ttl())
			.isEqualTo(AnthropicCacheTtl.FIVE_MINUTES.getValue());

		// Current user question also must not get cache_control in agentic strategy
		var lastUserMessage = request.messages().get(5);
		assertThat(lastUserMessage.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(lastUserMessage.content()).hasSize(1);
		assertThat(lastUserMessage.content().get(0).cacheControl()).isNull();
	}

	@Test
	void agenticToolUseMovesBreakpointToLastStableAssistantWhenDeltaIsLargeEnough() {
		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("claude-3-7-sonnet-latest").build())
			.build();

		AnthropicCacheOptions cacheOptions = AnthropicCacheOptions.builder()
			.strategy(AnthropicCacheStrategy.AGENTIC_TOOL_USE)
			.messageTypeTtl(MessageType.ASSISTANT, AnthropicCacheTtl.FIVE_MINUTES)
			// Force a small threshold so the delta qualifies in test.
			.minCacheablePromptLength(50)
			.build();

		var promptOptions = AnthropicChatOptions.builder().cacheOptions(cacheOptions).build();

		String longAssistant = "A".repeat(200);
		List<Message> messages = List.of(new UserMessage("Initial request"),
				ToolResponseMessage.builder()
					.responses(List.of(new ToolResponseMessage.ToolResponse("call-1", "read_file", "{\"file\":\"x\"}")))
					.build(),
				AssistantMessage.builder().content(longAssistant).build(), new UserMessage("Current question"));

		var request = client.createRequest(new Prompt(messages, promptOptions), false);

		// TOOL_RESULT should not be cached (breakpoint moved to ASSISTANT)
		var toolResult = request.messages().get(1);
		assertThat(toolResult.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(toolResult.content()).hasSize(1);
		assertThat(toolResult.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TOOL_RESULT);
		assertThat(toolResult.content().get(0).cacheControl()).isNull();

		// ASSISTANT text should be cached
		var assistant = request.messages().get(2);
		assertThat(assistant.role()).isEqualTo(AnthropicApi.Role.ASSISTANT);
		assertThat(assistant.content()).hasSize(1);
		assertThat(assistant.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TEXT);
		assertThat(assistant.content().get(0).cacheControl()).isNotNull();
	}

	@Test
	void agenticToolUseNoCacheWhenNoToolAndNoStableAssistant() {
		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("claude-3-7-sonnet-latest").build())
			.build();

		AnthropicCacheOptions cacheOptions = AnthropicCacheOptions.builder()
			.strategy(AnthropicCacheStrategy.AGENTIC_TOOL_USE)
			.minCacheablePromptLength(50)
			.build();

		var promptOptions = AnthropicChatOptions.builder().cacheOptions(cacheOptions).build();

		// Only USER messages and ASSISTANT with toolCalls (no TOOL, no stable ASSISTANT)
		List<Message> messages = List.of(new UserMessage("Initial request"), AssistantMessage.builder()
			.content("I will call a tool")
			.toolCalls(
					List.of(new AssistantMessage.ToolCall("call-1", "tool_use", "get_weather", "{\"city\":\"Paris\"}")))
			.build(), new UserMessage("Current question"));

		var request = client.createRequest(new Prompt(messages, promptOptions), false);

		// No messages should have cache_control since there's no TOOL and no stable
		// ASSISTANT
		for (var msg : request.messages()) {
			for (var content : msg.content()) {
				assertThat(content.cacheControl()).isNull();
			}
		}
	}

	@Test
	void agenticToolUseFullAgenticLoopScenario() {
		var client = AnthropicChatModel.builder()
			.anthropicApi(AnthropicApi.builder().apiKey("TEST").build())
			.defaultOptions(AnthropicChatOptions.builder().model("claude-3-7-sonnet-latest").build())
			.build();

		AnthropicCacheOptions cacheOptions = AnthropicCacheOptions.builder()
			.strategy(AnthropicCacheStrategy.AGENTIC_TOOL_USE)
			.messageTypeTtl(MessageType.TOOL, AnthropicCacheTtl.FIVE_MINUTES)
			// High threshold so breakpoint stays on last TOOL
			.minCacheablePromptLength(100_000)
			.build();

		var promptOptions = AnthropicChatOptions.builder().cacheOptions(cacheOptions).build();

		// Realistic agentic loop: USER -> ASSISTANT(toolCalls) -> TOOL ->
		// ASSISTANT(toolCalls) -> TOOL
		List<Message> messages = List.of(new UserMessage("Analyze the codebase and find bugs"),
				AssistantMessage.builder()
					.content("I'll start by reading the main file")
					.toolCalls(List
						.of(new AssistantMessage.ToolCall("call-1", "tool_use", "read_file", "{\"path\":\"main.py\"}")))
					.build(),
				ToolResponseMessage.builder()
					.responses(List
						.of(new ToolResponseMessage.ToolResponse("call-1", "read_file", "def main(): print('hello')")))
					.build(),
				AssistantMessage.builder()
					.content("Now I'll check the tests")
					.toolCalls(List
						.of(new AssistantMessage.ToolCall("call-2", "tool_use", "read_file", "{\"path\":\"test.py\"}")))
					.build(),
				ToolResponseMessage.builder()
					.responses(List.of(new ToolResponseMessage.ToolResponse("call-2", "read_file",
							"def test_main(): assert True")))
					.build());

		var request = client.createRequest(new Prompt(messages, promptOptions), false);

		// First USER - no cache
		var userMsg = request.messages().get(0);
		assertThat(userMsg.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(userMsg.content().get(0).cacheControl()).isNull();

		// First ASSISTANT with toolCalls - no cache on tool_use blocks
		var assistant1 = request.messages().get(1);
		assertThat(assistant1.role()).isEqualTo(AnthropicApi.Role.ASSISTANT);
		assertThat(assistant1.content()).anyMatch(cb -> cb.type() == AnthropicApi.ContentBlock.Type.TOOL_USE);
		for (var content : assistant1.content()) {
			if (content.type() == AnthropicApi.ContentBlock.Type.TOOL_USE) {
				assertThat(content.cacheControl()).isNull();
			}
		}

		// First TOOL_RESULT - no cache (not the last one)
		var tool1 = request.messages().get(2);
		assertThat(tool1.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(tool1.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TOOL_RESULT);
		assertThat(tool1.content().get(0).cacheControl()).isNull();

		// Second ASSISTANT with toolCalls - no cache
		var assistant2 = request.messages().get(3);
		assertThat(assistant2.role()).isEqualTo(AnthropicApi.Role.ASSISTANT);
		for (var content : assistant2.content()) {
			if (content.type() == AnthropicApi.ContentBlock.Type.TOOL_USE) {
				assertThat(content.cacheControl()).isNull();
			}
		}

		// Last TOOL_RESULT - SHOULD have cache_control
		var tool2 = request.messages().get(4);
		assertThat(tool2.role()).isEqualTo(AnthropicApi.Role.USER);
		assertThat(tool2.content().get(0).type()).isEqualTo(AnthropicApi.ContentBlock.Type.TOOL_RESULT);
		assertThat(tool2.content().get(0).cacheControl()).isNotNull();
		assertThat(tool2.content().get(0).cacheControl().ttl()).isEqualTo(AnthropicCacheTtl.FIVE_MINUTES.getValue());
	}

}
