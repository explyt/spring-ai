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

package org.springframework.ai.anthropic.api.utils;

import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.anthropic.api.AnthropicCacheOptions;
import org.springframework.ai.anthropic.api.AnthropicCacheStrategy;
import org.springframework.ai.anthropic.api.AnthropicCacheTtl;
import org.springframework.ai.anthropic.api.AnthropicCacheType;
import org.springframework.ai.chat.messages.MessageType;

/**
 * Resolves cache eligibility for messages based on the provided
 * {@link AnthropicCacheOptions}.
 *
 * Note: Tool definition messages are always considered for caching if the strategy
 * includes system messages. The minimum content length check is not applied to tool
 * definition messages.
 *
 * @author Austin Dase
 * @author Soby Chacko
 * @since 1.1.0
 **/
public class CacheEligibilityResolver {

	private static final Logger logger = LoggerFactory.getLogger(CacheEligibilityResolver.class);

	// Tool definition messages are always considered for caching if the strategy
	// includes system messages.
	private static final MessageType TOOL_DEFINITION_MESSAGE_TYPE = MessageType.SYSTEM;

	private final CacheBreakpointTracker cacheBreakpointTracker = new CacheBreakpointTracker();

	private final AnthropicCacheType anthropicCacheType = AnthropicCacheType.EPHEMERAL;

	private final AnthropicCacheStrategy cacheStrategy;

	private final Map<MessageType, AnthropicCacheTtl> messageTypeTtl;

	private final Map<MessageType, Integer> messageTypeMinContentLengths;

	private final Function<String, Integer> contentLengthFunction;

	private final Function<String, Integer> tokenLengthFunction;

	private final Set<MessageType> cacheEligibleMessageTypes;

	private final Integer minCacheablePromptLength;

	public CacheEligibilityResolver(AnthropicCacheStrategy cacheStrategy,
			Map<MessageType, AnthropicCacheTtl> messageTypeTtl, Map<MessageType, Integer> messageTypeMinContentLengths,
			Function<String, Integer> contentLengthFunction, Function<String, Integer> tokenLengthFunction,
			Set<MessageType> cacheEligibleMessageTypes, Integer minCacheablePromptLength) {
		this.cacheStrategy = cacheStrategy;
		this.messageTypeTtl = messageTypeTtl;
		this.messageTypeMinContentLengths = messageTypeMinContentLengths;
		this.contentLengthFunction = contentLengthFunction;
		this.tokenLengthFunction = tokenLengthFunction;
		this.cacheEligibleMessageTypes = cacheEligibleMessageTypes;
		this.minCacheablePromptLength = minCacheablePromptLength;
	}

	public static CacheEligibilityResolver from(AnthropicCacheOptions anthropicCacheOptions) {
		AnthropicCacheStrategy strategy = anthropicCacheOptions.getStrategy();
		return new CacheEligibilityResolver(strategy, anthropicCacheOptions.getMessageTypeTtl(),
				anthropicCacheOptions.getMessageTypeMinContentLengths(),
				anthropicCacheOptions.getContentLengthFunction(), anthropicCacheOptions.getTokenLengthFunction(),
				extractEligibleMessageTypes(strategy), anthropicCacheOptions.getMinCacheablePromptLength());
	}

	private static Set<MessageType> extractEligibleMessageTypes(AnthropicCacheStrategy anthropicCacheStrategy) {
		return switch (anthropicCacheStrategy) {
			case NONE -> Set.of();
			case SYSTEM_ONLY, SYSTEM_AND_TOOLS -> Set.of(MessageType.SYSTEM);
			case TOOLS_ONLY -> Set.of(); // No message types cached, only tool definitions
			// For CONVERSATION_HISTORY and AGENTIC_TOOL_USE, all message types are
			// potentially eligible for caching. However, for AGENTIC_TOOL_USE the actual
			// cache application is controlled by agenticBreakpointIndices in
			// buildMessages(),
			// which selects only specific messages (last TOOL or stable ASSISTANT) for
			// breakpoints.
			case CONVERSATION_HISTORY, AGENTIC_TOOL_USE -> Set.of(MessageType.values());
		};
	}

	public AnthropicApi.ChatCompletionRequest.CacheControl resolve(MessageType messageType, String content) {
		Integer length = this.contentLengthFunction.apply(content);
		if (this.cacheStrategy == AnthropicCacheStrategy.NONE || !this.cacheEligibleMessageTypes.contains(messageType)
				|| length == null || length < this.messageTypeMinContentLengths.get(messageType)
				|| this.cacheBreakpointTracker.allBreakpointsAreUsed()) {
			logger.debug(
					"Caching not enabled for messageType={}, contentLength={}, minContentLength={}, cacheStrategy={}, usedBreakpoints={}",
					messageType, length, this.messageTypeMinContentLengths.get(messageType), this.cacheStrategy,
					this.cacheBreakpointTracker.getCount());
			return null;
		}

		AnthropicCacheTtl anthropicCacheTtl = this.messageTypeTtl.get(messageType);

		logger.debug("Caching enabled for messageType={}, ttl={}", messageType, anthropicCacheTtl);

		return this.anthropicCacheType.cacheControl(anthropicCacheTtl.getValue());
	}

	public AnthropicApi.ChatCompletionRequest.CacheControl resolveToolCacheControl() {
		// Tool definitions are cache-eligible for TOOLS_ONLY, SYSTEM_AND_TOOLS,
		// CONVERSATION_HISTORY, and AGENTIC_TOOL_USE strategies. SYSTEM_ONLY caches only
		// system messages, relying on Anthropic's cache hierarchy to implicitly cache
		// tools.
		if (this.cacheStrategy != AnthropicCacheStrategy.TOOLS_ONLY
				&& this.cacheStrategy != AnthropicCacheStrategy.SYSTEM_AND_TOOLS
				&& this.cacheStrategy != AnthropicCacheStrategy.CONVERSATION_HISTORY
				&& this.cacheStrategy != AnthropicCacheStrategy.AGENTIC_TOOL_USE) {
			logger.debug("Caching not enabled for tool definition, cacheStrategy={}", this.cacheStrategy);
			return null;
		}

		if (this.cacheBreakpointTracker.allBreakpointsAreUsed()) {
			logger.debug("Caching not enabled for tool definition, usedBreakpoints={}",
					this.cacheBreakpointTracker.getCount());
			return null;
		}

		AnthropicCacheTtl anthropicCacheTtl = this.messageTypeTtl.get(TOOL_DEFINITION_MESSAGE_TYPE);

		logger.debug("Caching enabled for tool definition, ttl={}", anthropicCacheTtl);

		return this.anthropicCacheType.cacheControl(anthropicCacheTtl.getValue());
	}

	public boolean isCachingEnabled() {
		return this.cacheStrategy != AnthropicCacheStrategy.NONE;
	}

	public void useCacheBlock() {
		this.cacheBreakpointTracker.use();
	}

	public AnthropicCacheStrategy getStrategy() {
		return this.cacheStrategy;
	}

	public Function<String, Integer> getContentLengthFunction() {
		return this.contentLengthFunction;
	}

	public Function<String, Integer> getTokenLengthFunction() {
		return this.tokenLengthFunction;
	}

	public int getRemainingBreakpoints() {
		return 4 - this.cacheBreakpointTracker.getCount();
	}

	public Integer getMinCacheablePromptLength() {
		return this.minCacheablePromptLength;
	}

	public Map<MessageType, AnthropicCacheTtl> getMessageTypeTtl() {
		return this.messageTypeTtl;
	}

}
