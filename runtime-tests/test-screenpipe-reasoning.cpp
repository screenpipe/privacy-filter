// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
// Exercise the production shared sampler with a synthetic vocabulary, no model.
#include "../common/sampling.cpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    auto * budget = common_reasoning_budget_init(nullptr, {10}, {11}, {11}, 3, std::vector<llama_token>{10});
    auto * sampler = new common_sampler { {}, nullptr, budget,
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        ring_buffer<llama_token>(32), {}, {} };
    // The server replays the whole prompt for repetition penalties, including
    // earlier thinking blocks. This must not spend the NEW response's budget.
    for (auto token : {1, 10, 2, 2, 2, 11, 1, 10}) common_sampler_accept(sampler, token, false);
    assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_COUNTING);
    common_sampler_accept(sampler, 2, true);
    common_sampler_accept(sampler, 2, true);
    assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_COUNTING);
    auto * copy = common_sampler_clone(sampler);
    common_sampler_accept(sampler, 2, true);
    common_sampler_accept(copy, 2, true);
    assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_FORCING);
    assert(common_reasoning_budget_get_state(copy->rbudget) == REASONING_BUDGET_FORCING);
    common_sampler_accept(sampler, 11, true);
    assert(common_reasoning_budget_get_state(budget) == REASONING_BUDGET_DONE);
    common_sampler_free(copy);
    common_sampler_free(sampler);
    puts("Prompt replay and speculative clone retain the response reasoning budget");
}
