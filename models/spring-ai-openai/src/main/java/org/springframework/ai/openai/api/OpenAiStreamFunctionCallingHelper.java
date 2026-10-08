/*
 * Copyright 2023-2024 the original author or authors.
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

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk.ChunkChoice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ChatCompletionFunction;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.openai.api.OpenAiApi.LogProbs;
import org.springframework.ai.openai.api.OpenAiApi.Usage;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;

/**
 * Helper class to support Streaming function calling.
 *
 * It can merge the streamed ChatCompletionChunk in case of function calling message.
 *
 * @author Christian Tzolov
 * @author Thomas Vitale
 * @author Alexandros Pappas
 * @since 0.8.1
 */
public class OpenAiStreamFunctionCallingHelper {

	/**
	 * Merge the previous and current ChatCompletionChunk into a single one.
	 * @param previous the previous ChatCompletionChunk
	 * @param current the current ChatCompletionChunk
	 * @return the merged ChatCompletionChunk
	 */
	public ChatCompletionChunk merge(ChatCompletionChunk previous, ChatCompletionChunk current) {

		if (previous == null) {
			return current;
		}

		if (current == null) {
			return previous;
		}

		String id = (current.id() != null ? current.id() : previous.id());
		Long created = (current.created() != null ? current.created() : previous.created());
		String model = (current.model() != null ? current.model() : previous.model());
		String serviceTier = (current.serviceTier() != null ? current.serviceTier() : previous.serviceTier());
		String systemFingerprint = (current.systemFingerprint() != null ? current.systemFingerprint()
				: previous.systemFingerprint());
		String object = (current.object() != null ? current.object() : previous.object());
		Usage usage = (current.usage() != null ? current.usage() : previous.usage());

		ChunkChoice previousChoice0 = (CollectionUtils.isEmpty(previous.choices()) ? null : previous.choices().get(0));
		ChunkChoice currentChoice0 = (CollectionUtils.isEmpty(current.choices()) ? null : current.choices().get(0));

		ChunkChoice choice = merge(previousChoice0, currentChoice0);
		List<ChunkChoice> chunkChoices = choice == null ? List.of() : List.of(choice);
		return new ChatCompletionChunk(id, chunkChoices, created, model, serviceTier, systemFingerprint, object, usage);
	}

	private ChunkChoice merge(ChunkChoice previous, ChunkChoice current) {
		if (previous == null) {
			return current;
		}

		if (current == null) {
			return previous;
		}

		ChatCompletionFinishReason finishReason = (current.finishReason() != null ? current.finishReason()
				: previous.finishReason());
		Integer index = (current.index() != null ? current.index() : previous.index());

		ChatCompletionMessage message = merge(previous.delta(), current.delta());

		LogProbs logprobs = (current.logprobs() != null ? current.logprobs() : previous.logprobs());
		return new ChunkChoice(finishReason, index, message, logprobs);
	}

	private ChatCompletionMessage merge(ChatCompletionMessage previous, ChatCompletionMessage current) {
		String content = (current.content() != null ? current.content()
				: "" + ((previous.content() != null) ? previous.content() : ""));
		String reasoningContent = (current.getReasoningContent() != null ? current.getReasoningContent()
				: "" + ((previous.getReasoningContent() != null) ? previous.getReasoningContent() : ""));
		Role role = (current.role() != null ? current.role() : previous.role());
		role = (role != null ? role : Role.ASSISTANT); // default to ASSISTANT (if null
		String name = (current.name() != null ? current.name() : previous.name());
		String toolCallId = (current.toolCallId() != null ? current.toolCallId() : previous.toolCallId());
		String refusal = (current.refusal() != null ? current.refusal() : previous.refusal());
		ChatCompletionMessage.AudioOutput audioOutput = (current.audioOutput() != null ? current.audioOutput()
				: previous.audioOutput());
		List<ChatCompletionMessage.Annotation> annotations = (current.annotations() != null ? current.annotations()
				: previous.annotations());

		List<ToolCall> toolCalls = new ArrayList<>();
		ToolCall lastPreviousTooCall = null;
		if (previous.toolCalls() != null && !previous.toolCalls().isEmpty()) {
			lastPreviousTooCall = previous.toolCalls().get(previous.toolCalls().size() - 1);
			if (previous.toolCalls().size() > 1) {
				toolCalls.addAll(previous.toolCalls().subList(0, previous.toolCalls().size() - 1));
			}
		}
		if (current.toolCalls() != null && !current.toolCalls().isEmpty()) {
			if (current.toolCalls().size() > 1) {
				return new ChatCompletionMessage(content, role, name, toolCallId,
						mergeBatchedToolCalls(previous.toolCalls(), current.toolCalls()), refusal, audioOutput,
						annotations, reasoningContent, reasoningContent);
			}
			var currentToolCall = current.toolCalls().iterator().next();
			if (StringUtils.hasText(currentToolCall.id())) {
				if (lastPreviousTooCall != null) {
					toolCalls.add(lastPreviousTooCall);
				}
				toolCalls.add(currentToolCall);
			}
			else {
				toolCalls.add(merge(lastPreviousTooCall, currentToolCall));
			}
		}
		else {
			if (lastPreviousTooCall != null) {
				toolCalls.add(lastPreviousTooCall);
			}
		}
		return new ChatCompletionMessage(content, role, name, toolCallId, toolCalls, refusal, audioOutput, annotations,
				reasoningContent, reasoningContent);
	}

	/**
	 * Merges a delta that carries several tool-call fragments at once (some vLLM tool
	 * parsers batch parallel tool calls this way) into the accumulated tool calls. Used
	 * only for such deltas, which used to be rejected; a delta with a single fragment
	 * keeps the id-based merge above unchanged.
	 * <p>
	 * A fragment with an {@code index} and an {@code id} continues the tool call that has
	 * the same id and the same index; otherwise it starts a new tool call (this also
	 * covers upstreams that reuse one index for every call). A fragment with an
	 * {@code index} and no {@code id} continues the latest tool call with that index.
	 * Tool calls continued by single-fragment deltas carry no index (the merge above
	 * drops it); such a call owns a fragment only if it is the sole possible owner.
	 * Ambiguous fragments are rejected, as before, rather than silently corrupting
	 * another call. Fragments without an index follow the id-based rule of the
	 * single-fragment merge.
	 */
	private List<ToolCall> mergeBatchedToolCalls(List<ToolCall> previous, List<ToolCall> current) {
		List<ToolCall> toolCalls = new ArrayList<>();
		if (previous != null) {
			for (ToolCall toolCall : previous) {
				if (toolCall != null) {
					toolCalls.add(toolCall);
				}
			}
		}
		for (ToolCall fragment : current) {
			if (fragment != null) {
				mergeBatchedToolCallFragment(toolCalls, fragment);
			}
		}
		return toolCalls;
	}

	private void mergeBatchedToolCallFragment(List<ToolCall> toolCalls, ToolCall current) {
		int target = -1;
		if (current.index() == null) {
			// No index: the id-based rule of the single-fragment merge.
			if (!StringUtils.hasText(current.id()) && !toolCalls.isEmpty()) {
				target = toolCalls.size() - 1;
			}
		}
		else if (StringUtils.hasText(current.id())) {
			int sameId = lastIndexWhere(toolCalls, toolCall -> current.id().equals(toolCall.id()));
			if (sameId >= 0 && toolCalls.get(sameId).index() == null) {
				// Continuation or a new call reusing the id: cannot tell without the
				// index.
				throw cannotAttribute(current);
			}
			if (sameId >= 0 && current.index().equals(toolCalls.get(sameId).index())) {
				target = sameId;
			}
		}
		else {
			// Candidates are the calls with this index and the calls whose index is
			// unknown. The owner is the latest candidate if its index matches, or the
			// only candidate; anything else is ambiguous.
			Predicate<ToolCall> candidate = toolCall -> toolCall.index() == null
					|| current.index().equals(toolCall.index());
			target = lastIndexWhere(toolCalls, candidate);
			if (target < 0
					|| (toolCalls.get(target).index() == null && toolCalls.stream().filter(candidate).count() > 1)) {
				throw cannotAttribute(current);
			}
		}

		if (target >= 0) {
			toolCalls.set(target, mergeKeepingIndex(toolCalls.get(target), withFunction(current)));
		}
		else {
			toolCalls.add(withFunction(current));
		}
	}

	private static IllegalStateException cannotAttribute(ToolCall fragment) {
		return new IllegalStateException("Cannot attribute a streamed tool call fragment (index " + fragment.index()
				+ ", id " + fragment.id() + ") to a single tool call!");
	}

	private static int lastIndexWhere(List<ToolCall> toolCalls, Predicate<ToolCall> predicate) {
		for (int i = toolCalls.size() - 1; i >= 0; i--) {
			if (predicate.test(toolCalls.get(i))) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * A continuation fragment may omit {@code function} (e.g. carry only the type); an
	 * empty function keeps the accumulated name and arguments intact.
	 */
	private static ToolCall withFunction(ToolCall toolCall) {
		return (toolCall.function() != null ? toolCall : new ToolCall(toolCall.index(), toolCall.id(), toolCall.type(),
				new ChatCompletionFunction(null, null)));
	}

	/**
	 * Same as {@link #merge(ToolCall, ToolCall)}, but keeps the first seen {@code index}
	 * so that later fragments of a batch can still be routed to this tool call.
	 */
	private ToolCall mergeKeepingIndex(ToolCall previous, ToolCall current) {
		ToolCall merged = merge(previous, current);
		Integer index = (previous.index() != null ? previous.index() : current.index());
		return new ToolCall(index, merged.id(), merged.type(), merged.function());
	}

	private ToolCall merge(ToolCall previous, ToolCall current) {
		if (previous == null) {
			return current;
		}
		String id = (StringUtils.hasText(current.id()) ? current.id() : previous.id());
		String type = (current.type() != null ? current.type() : previous.type());
		ChatCompletionFunction function = merge(previous.function(), current.function());
		return new ToolCall(id, type, function);
	}

	private ChatCompletionFunction merge(ChatCompletionFunction previous, ChatCompletionFunction current) {
		if (previous == null) {
			return current;
		}
		String name = (StringUtils.hasText(current.name()) ? current.name() : previous.name());
		StringBuilder arguments = new StringBuilder();
		if (previous.arguments() != null) {
			arguments.append(previous.arguments());
		}
		if (current.arguments() != null) {
			arguments.append(current.arguments());
		}
		return new ChatCompletionFunction(name, arguments.toString());
	}

	/**
	 * @param chatCompletion the ChatCompletionChunk to check
	 * @return true if the ChatCompletionChunk is a streaming tool function call.
	 */
	public boolean isStreamingToolFunctionCall(ChatCompletionChunk chatCompletion) {

		if (chatCompletion == null || CollectionUtils.isEmpty(chatCompletion.choices())) {
			return false;
		}

		var choice = chatCompletion.choices().get(0);
		if (choice == null || choice.delta() == null) {
			return false;
		}
		return !CollectionUtils.isEmpty(choice.delta().toolCalls());
	}

	/**
	 * @param chatCompletion the ChatCompletionChunk to check
	 * @return true if the ChatCompletionChunk is a streaming tool function call and it is
	 * the last one.
	 */
	public boolean isStreamingToolFunctionCallFinish(ChatCompletionChunk chatCompletion) {

		if (chatCompletion == null || CollectionUtils.isEmpty(chatCompletion.choices())) {
			return false;
		}

		var choice = chatCompletion.choices().get(0);
		if (choice == null || choice.delta() == null) {
			return false;
		}
		return choice.finishReason() == ChatCompletionFinishReason.TOOL_CALLS;
	}

	/**
	 * Convert the ChatCompletionChunk into a ChatCompletion. The Usage is set to null.
	 * @param chunk the ChatCompletionChunk to convert
	 * @return the ChatCompletion
	 */
	public ChatCompletion chunkToChatCompletion(ChatCompletionChunk chunk) {
		List<Choice> choices = chunk.choices()
			.stream()
			.map(chunkChoice -> new Choice(chunkChoice.finishReason(), chunkChoice.index(), chunkChoice.delta(),
					chunkChoice.logprobs()))
			.toList();

		return new OpenAiApi.ChatCompletion(chunk.id(), choices, chunk.created(), chunk.model(), chunk.serviceTier(),
				chunk.systemFingerprint(), "chat.completion", null);
	}

}
// ---
