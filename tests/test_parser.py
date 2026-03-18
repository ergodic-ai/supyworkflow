"""Tests for cell parser and dependency analysis."""

import pytest

from supyworkflow.parser import Cell, build_dependency_graph, parse_cells


class TestParseCells:
    def test_single_cell_no_markers(self):
        source = "x = 1\ny = x + 2"
        cells = parse_cells(source)
        assert len(cells) == 1
        assert cells[0].label == "setup"
        assert "x" in cells[0].writes
        assert "y" in cells[0].writes

    def test_basic_split(self):
        source = """
# --- Fetch data
orders = shopify.get_orders()

# --- Process
total = sum(o["price"] for o in orders)

# --- Report
gmail.send_email(body=str(total))
"""
        cells = parse_cells(source)
        assert len(cells) == 3
        assert cells[0].label == "Fetch data"
        assert cells[1].label == "Process"
        assert cells[2].label == "Report"

    def test_empty_cells_skipped(self):
        source = """
# --- First
x = 1

# --- Empty

# --- Third
y = x
"""
        cells = parse_cells(source)
        assert len(cells) == 2
        assert cells[0].label == "First"
        assert cells[1].label == "Third"

    def test_reads_and_writes(self):
        source = """
# --- Setup
data = [1, 2, 3]
total = 0

# --- Process
total = sum(data)
result = total * 2
"""
        cells = parse_cells(source)
        assert "data" in cells[0].writes
        assert "total" in cells[0].writes
        assert "data" in cells[1].reads
        assert "total" in cells[1].writes
        assert "result" in cells[1].writes

    def test_for_loop_writes(self):
        source = "for item in items:\n    x = item + 1"
        cells = parse_cells(source)
        assert "item" in cells[0].writes
        assert "x" in cells[0].writes
        assert "items" in cells[0].reads

    def test_augmented_assign(self):
        source = "total += value"
        cells = parse_cells(source)
        assert "total" in cells[0].writes
        assert "value" in cells[0].reads

    def test_syntax_error_graceful(self):
        source = "this is not valid python @@!!"
        cells = parse_cells(source)
        assert len(cells) == 1
        assert cells[0].reads == set()
        assert cells[0].writes == set()


class TestDependencyGraph:
    def test_linear_chain(self):
        source = """
# --- A
x = 1

# --- B
y = x + 1

# --- C
z = y + 1
"""
        cells = parse_cells(source)
        graph = build_dependency_graph(cells)
        assert graph[0] == set()
        assert graph[1] == {0}
        assert graph[2] == {1}

    def test_diamond_dependency(self):
        source = """
# --- Fetch
data = fetch()

# --- Branch A
a = process_a(data)

# --- Branch B
b = process_b(data)

# --- Merge
result = combine(a, b)
"""
        cells = parse_cells(source)
        graph = build_dependency_graph(cells)
        assert graph[0] == set()
        assert graph[1] == {0}  # depends on data
        assert graph[2] == {0}  # depends on data
        assert graph[3] == {1, 2}  # depends on a and b

    def test_no_deps(self):
        source = """
# --- A
x = 1

# --- B
y = 2
"""
        cells = parse_cells(source)
        graph = build_dependency_graph(cells)
        assert graph[0] == set()
        assert graph[1] == set()

    def test_depends_on_property(self):
        source = """
# --- Setup
x = 1
y = 2

# --- Use
z = x + y + z_prev
"""
        cells = parse_cells(source)
        # Cell 1 reads x, y, z_prev but writes z
        # depends_on = reads - writes = {x, y, z_prev}
        assert "z_prev" in cells[1].depends_on
        assert "x" in cells[1].depends_on
