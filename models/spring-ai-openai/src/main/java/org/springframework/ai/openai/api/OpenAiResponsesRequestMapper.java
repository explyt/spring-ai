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

package org.springframework.ai.openai.api;

import java.util.List;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest;

/**
 * Builds an OpenAI <b>Responses API</b> request body from a Chat Completions
 * {@link ChatCompletionRequest}, so that {@link OpenAiApi} configured with
 * {@link OpenAiWireApi#RESPONSES} serves the same typed requests as the Chat Completions
 * endpoint. The request is stateless: {@code store} is always {@code false} and no
 * {@code previous_response_id} is sent, so every call carries the whole conversation.
 * Chat Completions parameters without a Responses equivalent are rejected with an
 * {@link IllegalArgumentException} rather than dropped silently.
 */
public final class OpenAiResponsesRequestMapper {

	private static final List<String> UNSUPPORTED_PARAMETERS = List.of("frequency_penalty", "presence_penalty",
			"logit_bias", "logprobs", "top_logprobs", "seed", "stop", "modalities", "audio", "web_search_options");

	private static final List<String> COPIED_PARAMETERS = List.of("model", "temperature", "top_p",
			"parallel_tool_calls", "metadata", "user", "prompt_cache_key", "service_tier");

	private OpenAiResponsesRequestMapper() {
	}

	/**
	 * Maps a Chat Completions request to a Responses request body.
	 * @param request the typed Chat Completions request.
	 * @param reasoningSummary the {@code reasoning.summary} to request, or {@code null}
	 * to request none.
	 * @return the Responses request body.
	 */
	public static ObjectNode toResponsesRequest(ChatCompletionRequest request, String reasoningSummary) {
		JsonNode chat = ModelOptionsUtils.JSON_MAPPER.valueToTree(request);
		rejectUnsupportedParameters(chat);
		ObjectNode body = ModelOptionsUtils.JSON_MAPPER.createObjectNode();
		COPIED_PARAMETERS.forEach(name -> putIfPresent(body, name, chat.path(name)));
		body.put("stream", chat.path("stream").asBoolean(false));
		body.put("store", false);
		putIfPresent(body, "max_output_tokens",
				firstPresent(chat.path("max_completion_tokens"), chat.path("max_tokens")));
		body.set("input", input(chat.path("messages")));
		putTools(body, chat.path("tools"));
		putToolChoice(body, chat.path("tool_choice"));
		putReasoning(body, chat.path("reasoning_effort"), reasoningSummary);
		putText(body, chat.path("response_format"), chat.path("verbosity"));
		return body;
	}

	private static void rejectUnsupportedParameters(JsonNode chat) {
		for (String name : UNSUPPORTED_PARAMETERS) {
			if (isSet(chat.path(name))) {
				throw unsupported("'" + name + "'");
			}
		}
		if (chat.path("n").asInt(1) != 1) {
			throw unsupported("'n' other than 1");
		}
	}

	private static boolean isSet(JsonNode value) {
		if (value.isMissingNode() || value.isNull()) {
			return false;
		}
		if (value.isBoolean()) {
			return value.booleanValue();
		}
		return !(value.isContainer() && value.isEmpty());
	}

	private static IllegalArgumentException unsupported(String what) {
		return new IllegalArgumentException(what + " is not supported by the OpenAI Responses API");
	}

	private static ArrayNode input(JsonNode messages) {
		ArrayNode input = ModelOptionsUtils.JSON_MAPPER.createArrayNode();
		for (JsonNode message : messages) {
			String role = message.path("role").asText("");
			switch (role) {
				case "system", "user" -> putInputContent(messageItem(input, role), message.path("content"));
				case "assistant" -> addAssistantItems(input, message);
				case "tool" -> addToolOutput(input, message);
				default -> throw new IllegalArgumentException("Unsupported message role: " + role);
			}
		}
		return input;
	}

	private static ObjectNode messageItem(ArrayNode input, String role) {
		return input.addObject().put("type", "message").put("role", role);
	}

	private static void putInputContent(ObjectNode item, JsonNode content) {
		if (!content.isArray()) {
			item.put("content", content.asText(""));
			return;
		}
		ArrayNode parts = item.putArray("content");
		for (JsonNode part : content) {
			String type = part.path("type").asText("");
			switch (type) {
				case "text" -> parts.addObject().put("type", "input_text").put("text", part.path("text").asText(""));
				case "image_url" -> parts.addObject()
					.put("type", "input_image")
					.put("image_url", part.path("image_url").path("url").asText(""))
					.put("detail", part.path("image_url").path("detail").asText("auto"));
				case "file" -> {
					ObjectNode file = parts.addObject().put("type", "input_file");
					putIfPresent(file, "filename", part.path("file").path("filename"));
					putIfPresent(file, "file_data", part.path("file").path("file_data"));
				}
				case "input_audio" -> throw unsupported("Audio input");
				default -> throw new IllegalArgumentException("Unsupported content part type: " + type);
			}
		}
	}

	private static void addAssistantItems(ArrayNode input, JsonNode message) {
		if (isSet(message.path("audio"))) {
			throw unsupported("Assistant audio");
		}
		String text = textOf(message.path("content"));
		if (!text.isEmpty()) {
			messageItem(input, "assistant").put("content", text);
		}
		for (JsonNode toolCall : message.path("tool_calls")) {
			JsonNode function = toolCall.path("function");
			String arguments = function.path("arguments").asText("");
			input.addObject()
				.put("type", "function_call")
				.put("call_id", toolCall.path("id").asText(""))
				.put("name", function.path("name").asText(""))
				.put("arguments", arguments.isBlank() ? "{}" : arguments);
		}
	}

	private static void addToolOutput(ArrayNode input, JsonNode message) {
		input.addObject()
			.put("type", "function_call_output")
			.put("call_id", message.path("tool_call_id").asText(""))
			.put("output", textOf(message.path("content")));
	}

	private static String textOf(JsonNode content) {
		if (!content.isArray()) {
			return content.asText("");
		}
		StringBuilder text = new StringBuilder();
		content.forEach(part -> text.append(part.path("text").asText("")));
		return text.toString();
	}

	private static void putTools(ObjectNode body, JsonNode tools) {
		if (!tools.isArray() || tools.isEmpty()) {
			return;
		}
		ArrayNode target = body.putArray("tools");
		for (JsonNode tool : tools) {
			JsonNode function = tool.path("function");
			ObjectNode item = target.addObject().put("type", "function").put("name", function.path("name").asText(""));
			putIfPresent(item, "description", function.path("description"));
			putIfPresent(item, "parameters", function.path("parameters"));
			item.put("strict", function.path("strict").asBoolean(false));
		}
	}

	private static void putToolChoice(ObjectNode body, JsonNode toolChoice) {
		JsonNode functionName = toolChoice.path("function").path("name");
		if (functionName.isString()) {
			body.putObject("tool_choice").put("type", "function").put("name", functionName.asText());
		}
		else {
			putIfPresent(body, "tool_choice", toolChoice);
		}
	}

	private static void putReasoning(ObjectNode body, JsonNode effort, String summary) {
		if (!effort.isString() && summary == null) {
			return;
		}
		ObjectNode reasoning = body.putObject("reasoning");
		putIfPresent(reasoning, "effort", effort);
		if (summary != null) {
			reasoning.put("summary", summary);
		}
	}

	private static void putText(ObjectNode body, JsonNode responseFormat, JsonNode verbosity) {
		ObjectNode format = textFormat(responseFormat);
		if (format == null && !verbosity.isString()) {
			return;
		}
		ObjectNode text = body.putObject("text");
		if (format != null) {
			text.set("format", format);
		}
		putIfPresent(text, "verbosity", verbosity);
	}

	private static ObjectNode textFormat(JsonNode responseFormat) {
		String type = responseFormat.path("type").asText(null);
		if (type == null) {
			return null;
		}
		ObjectNode format = ModelOptionsUtils.JSON_MAPPER.createObjectNode().put("type", type);
		if ("json_schema".equals(type)) {
			JsonNode jsonSchema = responseFormat.path("json_schema");
			putIfPresent(format, "name", jsonSchema.path("name"));
			putIfPresent(format, "schema", jsonSchema.path("schema"));
			putIfPresent(format, "strict", jsonSchema.path("strict"));
		}
		return format;
	}

	private static JsonNode firstPresent(JsonNode first, JsonNode second) {
		return first.isMissingNode() || first.isNull() ? second : first;
	}

	private static void putIfPresent(ObjectNode target, String name, JsonNode value) {
		if (!value.isMissingNode() && !value.isNull()) {
			target.set(name, value.deepCopy());
		}
	}

}
