---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	perspective_generation(perspective_generation)
	query_expansion(query_expansion)
	arxiv_search(arxiv_search)
	relevance_filter(relevance_filter)
	synthesis(synthesis)
	evaluator(evaluator)
	reviser(reviser)
	__end__([<p>__end__</p>]):::last
	__start__ --> perspective_generation;
	arxiv_search --> relevance_filter;
	evaluator -. &nbsp;end_process&nbsp; .-> __end__;
	evaluator -. &nbsp;needs_revision&nbsp; .-> reviser;
	perspective_generation --> query_expansion;
	query_expansion --> arxiv_search;
	relevance_filter --> synthesis;
	reviser --> evaluator;
	synthesis --> evaluator;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
