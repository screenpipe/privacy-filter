// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
// Exercise the production shared sampler with a synthetic vocabulary, no model.
#include "../common/sampling.cpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    auto * budget = common_reasoning_budget_init(nullptr, {10}, {11}, {11}, 3, std::vector<llama_token>{99, 10});
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
    // Closed thinking stays idle; a later open block restarts the budget.
    for (const auto & prefill : std::vector<std::vector<llama_token>>{{99, 10, 2, 11}, {99, 10, 2, 11, 99, 10}}) {
        auto * b = common_reasoning_budget_init(nullptr, {10}, {11}, {11}, 3, prefill);
        assert(common_reasoning_budget_get_state(b) == (prefill.back() == 10 ? REASONING_BUDGET_COUNTING : REASONING_BUDGET_IDLE));
        llama_sampler_free(b);
    }
    auto * zero = common_reasoning_budget_init(nullptr, {10}, {11}, {11}, 0, std::vector<llama_token>{99, 10});
    assert(common_reasoning_budget_get_state(zero) == REASONING_BUDGET_FORCING);
    llama_sampler_free(zero);
    // Cloning halfway through a multi-token delimiter or forced close must
    // retain its position, not restart the sequence and leak reasoning.
    auto * partial = common_reasoning_budget_init(nullptr, {10, 12}, {11, 13}, {11, 13}, 4, std::vector<llama_token>{});
    llama_sampler_accept(partial, 10);
    auto * partial_copy = llama_sampler_clone(partial);
    llama_sampler_accept(partial_copy, 12);
    assert(common_reasoning_budget_get_state(partial_copy) == REASONING_BUDGET_COUNTING);
    llama_sampler_accept(partial_copy, 11);
    auto * end_copy = llama_sampler_clone(partial_copy);
    llama_sampler_accept(end_copy, 13);
    assert(common_reasoning_budget_get_state(end_copy) == REASONING_BUDGET_DONE);
    llama_sampler_free(partial);
    llama_sampler_free(partial_copy);
    llama_sampler_free(end_copy);
    auto * forcing = common_reasoning_budget_init(nullptr, {10}, {11, 13}, {11, 13}, 0, std::vector<llama_token>{99, 10});
    llama_sampler_accept(forcing, 11);
    auto * forcing_copy = llama_sampler_clone(forcing);
    llama_sampler_accept(forcing_copy, 13);
    assert(common_reasoning_budget_get_state(forcing_copy) == REASONING_BUDGET_DONE);
    llama_sampler_free(forcing);
    llama_sampler_free(forcing_copy);
    puts("Prompt replay and speculative clone retain the response reasoning budget");
}
