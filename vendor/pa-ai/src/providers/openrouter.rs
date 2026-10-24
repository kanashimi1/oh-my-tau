//! OpenRouter transport (TS `stream.ts` `openrouter` api): the Responses API by default, chat
//! completions when `PI_OPENROUTER_RESPONSES=0`.

use crate::event_stream::AssistantMessageEventStream;
use crate::providers::openai_completions::{stream_openai_completions, OpenAICompletionsOptions};
use crate::providers::openai_responses::{stream_openai_responses, OpenAIResponsesOptions};
use crate::registry::Provider;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};

pub const API_OPENROUTER: &str = "openrouter";

/// Registry provider for the `openrouter` api.
pub struct OpenRouterProvider;

impl Provider for OpenRouterProvider {
    fn api(&self) -> &str {
        API_OPENROUTER
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        if crate::transport::TransportEnv::from_process().openrouter_responses {
            let options = options.map(|base| OpenAIResponsesOptions::from_base(base.clone()));
            stream_openai_responses(model, context, options.as_ref())
        } else {
            let options = options.map(|base| OpenAICompletionsOptions::from_base(base.clone()));
            stream_openai_completions(model, context, options.as_ref())
        }
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        crate::transport::stream_simple_for_api(model, context, options)
    }
}
