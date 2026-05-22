from reasoning.base import ReasoningLayer
from base_llm import BaseLLM

MAX_NODES = 8


class GraphReasoning(ReasoningLayer):
    def __init__(self, llm: BaseLLM, max_nodes: int = MAX_NODES):
        super().__init__(llm)
        self.max_nodes = max_nodes

    def _decompose(self, question: str) -> list:
        prompt = (
            f"Question: {question}\n\n"
            f"Break into 3-4 atomic sub-questions. Each must be answerable with "
            f"a number or fact.\n"
            f"Format:\nNode 1: <sub-question>\nNode 2: <sub-question>\n"
            f"Node 3: <sub-question>"
        )
        response = self.llm.call(prompt)
        nodes = []
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("node"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    nodes.append({
                        "id": len(nodes),
                        "question": parts[1].strip(),
                        "answer": None,
                        "depends_on": [],
                        "resolved": False,
                    })
        return nodes

    def _infer_dependencies(self, question: str, nodes: list) -> list:
        node_list = "\n".join(
            f"Node {n['id']}: {n['question']}" for n in nodes
        )
        prompt = (
            f"Original: {question}\n\n"
            f"Sub-questions:\n{node_list}\n\n"
            f"For each node, list which other node IDs it depends on "
            f"(must be answered first). Use empty if independent.\n"
            f"Format one per line:\n"
            f"Node 0 deps: 1,2\nNode 1 deps:\n"
        )
        response = self.llm.call(prompt)
        dep_map = {n["id"]: [] for n in nodes}

        for line in response.splitlines():
            line = line.strip().lower()
            if not line.startswith("node"):
                continue
            try:
                head, rest = line.split("deps:", 1)
                nid = int("".join(c for c in head if c.isdigit()))
                deps = [
                    int(x.strip())
                    for x in rest.replace(",", " ").split()
                    if x.strip().isdigit()
                ]
                if nid in dep_map:
                    dep_map[nid] = [d for d in deps if d != nid and d < len(nodes)]
            except ValueError:
                continue

        for n in nodes:
            n["depends_on"] = dep_map.get(n["id"], [])
        return nodes

    def _detect_cycles(self, nodes: list) -> list:
        cycles = []
        visited = set()
        stack = set()
        id_to_node = {n["id"]: n for n in nodes}

        def dfs(nid, path):
            if nid in stack:
                cycle_start = path.index(nid)
                cycles.append(path[cycle_start:] + [nid])
                return
            if nid in visited:
                return
            visited.add(nid)
            stack.add(nid)
            path.append(nid)
            node = id_to_node.get(nid)
            if node:
                for dep in node.get("depends_on", []):
                    dfs(dep, path)
            path.pop()
            stack.remove(nid)

        for n in nodes:
            if n["id"] not in visited:
                dfs(n["id"], [])
        return cycles

    def _break_cycles(self, nodes: list) -> list:
        cycles = self._detect_cycles(nodes)
        for cycle in cycles:
            fix_id = cycle[0]
            for n in nodes:
                if n["id"] == fix_id:
                    cycle_set = set(cycle)
                    n["depends_on"] = [
                        d for d in n.get("depends_on", []) if d not in cycle_set
                    ]
        return nodes

    def _topological_order(self, nodes: list) -> list:
        in_degree = {n["id"]: 0 for n in nodes}
        for n in nodes:
            for d in n.get("depends_on", []):
                if d in in_degree:
                    in_degree[n["id"]] += 1

        queue = [n for n in nodes if in_degree[n["id"]] == 0]
        queue.sort(
            key=lambda n: -sum(
                1 for other in nodes if n["id"] in other.get("depends_on", [])
            )
        )
        ordered = []
        remaining = {n["id"]: n for n in nodes}

        while queue:
            queue.sort(
                key=lambda n: -sum(
                    1 for o in nodes if n["id"] in o.get("depends_on", [])
                )
            )
            node = queue.pop(0)
            ordered.append(node)
            del remaining[node["id"]]
            for n in list(remaining.values()):
                if node["id"] in n.get("depends_on", []):
                    n["depends_on"] = [
                        d for d in n["depends_on"] if d != node["id"]
                    ]
                    in_degree[n["id"]] = max(0, in_degree[n["id"]] - 1)
                    if in_degree[n["id"]] == 0:
                        queue.append(n)

        ordered.extend(remaining.values())
        return ordered

    def _resolve_node(self, node: dict, resolved_nodes: list) -> dict:
        context = ""
        deps = []
        if resolved_nodes:
            context = "Established facts (use these values):\n"
            for r in resolved_nodes:
                if r["id"] in node.get("depends_on", []) or not node.get("depends_on"):
                    context += f"  [{r['id']}] {r['question']} -> {r['answer']}\n"
                    deps.append(r["id"])
            for r in resolved_nodes:
                if r["id"] not in deps and node.get("depends_on"):
                    if r["id"] in node["depends_on"]:
                        context += f"  [{r['id']}] {r['question']} -> {r['answer']}\n"
                        deps.append(r["id"])
            context += "\n"

        prompt = (
            f"{context}"
            f"Answer this sub-question precisely:\n{node['question']}\n\n"
            f"End with: Answer: <value>"
        )
        response = self.llm.call(prompt).strip()
        clean = response
        if "answer:" in response.lower():
            idx = response.lower().rfind("answer:")
            clean = response[idx + len("answer:") :].strip()

        node["answer"] = clean
        node["full_resp"] = response
        node["resolved"] = True
        if not node.get("depends_on"):
            node["depends_on"] = deps
        return node

    def _validate_consistency(self, question: str, resolved_nodes: list) -> tuple:
        facts = "\n".join(
            f"Q{n['id']}: {n['question']}\nA{n['id']}: {n['answer']}"
            for n in resolved_nodes
        )
        prompt = (
            f"Original question: {question}\n\n"
            f"Resolved sub-problems:\n{facts}\n\n"
            f"Does the final conclusion follow from these answers? "
            f"If yes: CONSISTENT\n"
            f"If no: INCONSISTENT\nPROBLEM NODES: <comma-separated ids>"
        )
        response = self.llm.call(prompt).strip().upper()
        if "INCONSISTENT" in response and "CONSISTENT" not in response.split("INCONSISTENT")[0]:
            problem_nodes = []
            if "PROBLEM NODES:" in response:
                idx = response.find("PROBLEM NODES:") + len("PROBLEM NODES:")
                tail = response[idx:].strip()
                for part in tail.split(","):
                    digits = "".join(c for c in part if c.isdigit())
                    if digits:
                        problem_nodes.append(int(digits))
            return False, problem_nodes
        return True, []

    def _can_conclude(self, question: str, resolved_nodes: list) -> str | None:
        facts = ""
        for n in resolved_nodes:
            facts += f"  [{n['id']}] Q: {n['question']}\n       A: {n['answer']}\n"

        prompt = (
            f"Original question: {question}\n\n"
            f"Derived facts:\n{facts}\n"
            f"Using ONLY these values, answer the original question.\n"
            f"If YES: Final Answer: <value>\nIf NO: NEED MORE INFO"
        )
        response = self.llm.call(prompt).strip()
        return self._extract_final_answer(response)

    def _generate_new_node(self, question: str, resolved_nodes: list) -> dict:
        facts = "\n".join(
            f"  [{n['id']}] {n['question']} -> {n['answer']}" for n in resolved_nodes
        )
        prompt = (
            f"Original: {question}\n\nFacts:\n{facts}\n\n"
            f"What single sub-question is still needed? One line only."
        )
        sub_q = self.llm.call(prompt).strip()
        return {
            "id": len(resolved_nodes),
            "question": sub_q,
            "answer": None,
            "depends_on": [n["id"] for n in resolved_nodes[-2:]],
            "resolved": False,
        }

    def solve(self, question: str) -> dict:
        self.llm.reset_stats()
        all_steps = []
        resolved = []
        final_answer = None

        nodes = self._decompose(question)
        nodes = self._infer_dependencies(question, nodes)
        nodes = self._break_cycles(nodes)
        order = self._topological_order(nodes)

        all_steps.append(f"[decompose] {len(nodes)} nodes, cycle-safe order")
        for n in order:
            all_steps.append(
                f"  Node {n['id']} deps={n.get('depends_on', [])}: "
                f"{n['question'][:60]}"
            )

        unresolved = order[:]
        while len(resolved) < self.max_nodes:
            for node in unresolved:
                deps_ready = all(
                    d < len(resolved) for d in node.get("depends_on", [])
                )
                if node.get("depends_on") and not deps_ready:
                    continue
                context_nodes = [
                    r for r in resolved if r["id"] in node.get("depends_on", [])
                ] or resolved
                node = self._resolve_node(node, context_nodes)
                resolved.append(node)
                all_steps.append(
                    f"[node {node['id']}] -> {node['answer'][:80]}"
                )

            unresolved = []
            final_answer = self._can_conclude(question, resolved)
            if final_answer:
                all_steps.append("[conclude] sufficient nodes")
                break

            if len(resolved) < self.max_nodes:
                new_node = self._generate_new_node(question, resolved)
                all_steps.append(
                    f"[expand] node {new_node['id']}: {new_node['question'][:60]}"
                )
                unresolved = [new_node]
            else:
                break

        is_consistent, problem_nodes = self._validate_consistency(
            question, resolved
        )
        all_steps.append(
            f"[consistency] {'OK' if is_consistent else 'FAIL'} "
            f"problems={problem_nodes}"
        )

        if not is_consistent and problem_nodes:
            for nid in problem_nodes:
                for i, n in enumerate(resolved):
                    if n["id"] == nid:
                        prior = [r for r in resolved[:i] if r["id"] != nid]
                        resolved[i] = self._resolve_node(dict(n), prior)
                        all_steps.append(f"[re-resolve] node {nid}")
            is_consistent, _ = self._validate_consistency(question, resolved)

        if not final_answer:
            facts = "\n".join(
                f"  [{n['id']}] {n['question']} -> {n['answer']}" for n in resolved
            )
            fallback = (
                f"Question: {question}\n\nFacts:\n{facts}\n\n"
                f"Final Answer: <value> (one line arithmetic then answer)"
            )
            resp = self.llm.call(fallback).strip()
            final_answer = self._extract_final_answer(resp) or resp
            all_steps.append(f"[fallback] {final_answer[:80]}")

        return {
            "answer": final_answer,
            "steps": all_steps,
            "graph": resolved,
            "is_consistent": is_consistent,
            "stats": self.llm.get_stats(),
        }
