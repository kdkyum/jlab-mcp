"""Unit tests for notebook.py — local only, no SLURM needed."""

import nbformat
import pytest

from jlab_mcp.notebook import NotebookManager


@pytest.fixture
def nb_manager():
    return NotebookManager()


@pytest.fixture
def nb_path(nb_manager, tmp_path):
    return nb_manager.create_notebook("test_nb", tmp_path)


class TestCreateNotebook:
    def test_creates_file(self, nb_path):
        assert nb_path.exists()
        assert nb_path.suffix == ".ipynb"

    def test_valid_nbformat(self, nb_path):
        nb = nbformat.read(str(nb_path), as_version=4)
        assert nb.nbformat == 4
        assert nb.cells == []

    def test_has_kernelspec(self, nb_path):
        nb = nbformat.read(str(nb_path), as_version=4)
        assert "kernelspec" in nb.metadata
        assert nb.metadata.kernelspec.name == "python3"


class TestAddCodeCell:
    def test_add_cell(self, nb_manager, nb_path):
        idx = nb_manager.add_code_cell(nb_path, "print('hello')")
        assert idx == 0
        nb = nb_manager.get_notebook(nb_path)
        assert len(nb.cells) == 1
        assert nb.cells[0].cell_type == "code"
        assert nb.cells[0].source == "print('hello')"

    def test_add_multiple_cells(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        nb_manager.add_code_cell(nb_path, "y = 2")
        idx = nb_manager.add_code_cell(nb_path, "z = 3")
        assert idx == 2
        assert nb_manager.get_cell_count(nb_path) == 3

    def test_add_cell_with_outputs(self, nb_manager, nb_path):
        outputs = [{"type": "text", "content": "hello\n"}]
        nb_manager.add_code_cell(nb_path, "print('hello')", outputs)
        nb = nb_manager.get_notebook(nb_path)
        assert len(nb.cells[0].outputs) == 1

    def test_add_cell_at_beginning(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "second")
        idx = nb_manager.add_code_cell(nb_path, "first", index=0)
        assert idx == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].source == "first"
        assert nb.cells[1].source == "second"

    def test_add_cell_at_middle(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "a")
        nb_manager.add_code_cell(nb_path, "c")
        idx = nb_manager.add_code_cell(nb_path, "b", index=1)
        assert idx == 1
        nb = nb_manager.get_notebook(nb_path)
        assert [c.source for c in nb.cells] == ["a", "b", "c"]

    def test_add_cell_at_end_explicit(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "a")
        idx = nb_manager.add_code_cell(nb_path, "b", index=1)
        assert idx == 1
        nb = nb_manager.get_notebook(nb_path)
        assert [c.source for c in nb.cells] == ["a", "b"]

    def test_add_cell_negative_index_rejected(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "a")
        with pytest.raises(IndexError):
            nb_manager.add_code_cell(nb_path, "bad", index=-2)

    def test_add_cell_out_of_range(self, nb_manager, nb_path):
        with pytest.raises(IndexError):
            nb_manager.add_code_cell(nb_path, "bad", index=5)


class TestAddMarkdownCell:
    def test_add_markdown(self, nb_manager, nb_path):
        idx = nb_manager.add_markdown_cell(nb_path, "# Title\nSome text")
        assert idx == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[0].source == "# Title\nSome text"

    def test_add_markdown_at_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "code cell")
        idx = nb_manager.add_markdown_cell(nb_path, "# Header", index=0)
        assert idx == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "code"

    def test_add_markdown_negative_index_rejected(self, nb_manager, nb_path):
        with pytest.raises(IndexError):
            nb_manager.add_markdown_cell(nb_path, "bad", index=-2)


class TestEditCell:
    def test_edit_by_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "old code")
        resolved = nb_manager.edit_cell(nb_path, 0, "new code")
        assert resolved == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].source == "new code"

    def test_edit_negative_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "cell 0")
        nb_manager.add_code_cell(nb_path, "cell 1")
        nb_manager.add_code_cell(nb_path, "cell 2")
        resolved = nb_manager.edit_cell(nb_path, -1, "edited last")
        assert resolved == 2
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[2].source == "edited last"

    def test_edit_with_outputs(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "old")
        outputs = [{"type": "text", "content": "new output\n"}]
        nb_manager.edit_cell(nb_path, 0, "new", outputs)
        nb = nb_manager.get_notebook(nb_path)
        assert len(nb.cells[0].outputs) == 1

    def test_edit_out_of_range(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "only cell")
        with pytest.raises(IndexError):
            nb_manager.edit_cell(nb_path, 5, "bad index")


class TestGetCellSource:
    def test_get_by_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "cell_0")
        nb_manager.add_code_cell(nb_path, "cell_1")
        assert nb_manager.get_cell_source(nb_path, 0) == "cell_0"
        assert nb_manager.get_cell_source(nb_path, 1) == "cell_1"

    def test_get_negative_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "first")
        nb_manager.add_code_cell(nb_path, "last")
        assert nb_manager.get_cell_source(nb_path, -1) == "last"
        assert nb_manager.get_cell_source(nb_path, -2) == "first"

    def test_get_out_of_range(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "only")
        with pytest.raises(IndexError):
            nb_manager.get_cell_source(nb_path, 5)

    def test_get_negative_out_of_range(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "only")
        with pytest.raises(IndexError):
            nb_manager.get_cell_source(nb_path, -3)


class TestUpdateCellOutputs:
    def test_updates_outputs_only(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "print('hi')")
        outputs = [{"type": "text", "content": "hi\n"}]
        resolved = nb_manager.update_cell_outputs(nb_path, 0, outputs)
        assert resolved == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].source == "print('hi')"  # source unchanged
        assert len(nb.cells[0].outputs) == 1

    def test_updates_negative_index(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "a")
        nb_manager.add_code_cell(nb_path, "b")
        outputs = [{"type": "text", "content": "output\n"}]
        resolved = nb_manager.update_cell_outputs(nb_path, -1, outputs)
        assert resolved == 1
        nb = nb_manager.get_notebook(nb_path)
        assert len(nb.cells[1].outputs) == 1
        assert len(nb.cells[0].outputs) == 0  # first cell untouched

    def test_out_of_range(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "only")
        with pytest.raises(IndexError):
            nb_manager.update_cell_outputs(nb_path, 5, [])


class TestGetCodeCells:
    def test_returns_code_cells(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        nb_manager.add_markdown_cell(nb_path, "# Header")
        nb_manager.add_code_cell(nb_path, "y = 2")
        result = nb_manager.get_code_cells(nb_path)
        assert len(result) == 2
        assert result[0] == {"index": 0, "source": "x = 1"}
        assert result[1] == {"index": 2, "source": "y = 2"}

    def test_empty_notebook(self, nb_manager, nb_path):
        result = nb_manager.get_code_cells(nb_path)
        assert result == []

    def test_skips_markdown(self, nb_manager, nb_path):
        nb_manager.add_markdown_cell(nb_path, "# Only markdown")
        result = nb_manager.get_code_cells(nb_path)
        assert result == []


class TestCleanNotebook:
    def test_ensures_cell_ids(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        nb = nb_manager.get_notebook(nb_path)
        # Remove id to test that clean_notebook adds one back
        del nb.cells[0]["id"]
        nb_manager.save_notebook(nb_path, nb)
        nb = nb_manager.get_notebook(nb_path)
        assert "id" in nb.cells[0]
        assert len(nb.cells[0]["id"]) > 0


class TestCopyNotebook:
    def test_copy_with_suffix(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        new_path = nb_manager.copy_notebook(nb_path, "_continued")
        assert new_path.exists()
        assert "_continued" in new_path.stem
        nb = nb_manager.get_notebook(new_path)
        assert len(nb.cells) == 1


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_content(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "import numpy as np")
        nb_manager.add_markdown_cell(nb_path, "# Analysis")
        nb_manager.add_code_cell(nb_path, "result = np.array([1, 2, 3])")

        nb = nb_manager.get_notebook(nb_path)
        assert len(nb.cells) == 3
        assert nb.cells[0].source == "import numpy as np"
        assert nb.cells[1].source == "# Analysis"
        assert nb.cells[2].source == "result = np.array([1, 2, 3])"


class TestCreateNotebookNoOverwrite:
    def test_same_name_gets_unique_path(self, nb_manager, tmp_path):
        p1 = nb_manager.create_notebook("analysis", tmp_path)
        p2 = nb_manager.create_notebook("analysis", tmp_path)
        p3 = nb_manager.create_notebook("analysis", tmp_path)
        assert p1 != p2 != p3
        assert p2.stem == "analysis_2"
        assert p3.stem == "analysis_3"

    def test_existing_content_preserved(self, nb_manager, tmp_path):
        p1 = nb_manager.create_notebook("precious", tmp_path)
        nb_manager.add_code_cell(p1, "important_result = 42")
        nb_manager.create_notebook("precious", tmp_path)
        assert nb_manager.get_cell_count(p1) == 1
        assert nb_manager.get_cell_source(p1, 0) == "important_result = 42"


class TestNonCodeCellGuards:
    def test_edit_cell_rejects_markdown(self, nb_manager, nb_path):
        nb_manager.add_markdown_cell(nb_path, "# heading")
        with pytest.raises(ValueError, match="markdown"):
            nb_manager.edit_cell(nb_path, 0, "x = 1")

    def test_update_outputs_rejects_markdown(self, nb_manager, nb_path):
        nb_manager.add_markdown_cell(nb_path, "# heading")
        with pytest.raises(ValueError, match="markdown"):
            nb_manager.update_cell_outputs(
                nb_path, 0, [{"type": "text", "content": "x"}]
            )

    def test_edit_markdown_still_works(self, nb_manager, nb_path):
        nb_manager.add_markdown_cell(nb_path, "# old")
        nb_manager.edit_markdown_cell(nb_path, 0, "# new")
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].source == "# new"
        # Notebook must validate (no stray outputs on the markdown cell)
        nbformat.validate(nb)


class TestCellIdTracking:
    def test_get_cell_id(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        cell_id = nb_manager.get_cell_id(nb_path, 0)
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0]["id"] == cell_id

    def test_update_by_id_survives_index_shift(self, nb_manager, nb_path):
        """Outputs reach the right cell even after cells are inserted
        before it (the scenario where positional indices go stale)."""
        nb_manager.add_code_cell(nb_path, "target = 1")
        cell_id = nb_manager.get_cell_id(nb_path, 0)
        # Shift the target cell from index 0 to index 2
        nb_manager.add_code_cell(nb_path, "inserted_a", index=0)
        nb_manager.add_markdown_cell(nb_path, "# inserted_b", index=0)

        idx = nb_manager.update_cell_outputs_by_id(
            nb_path, cell_id, [{"type": "text", "content": "out\n"}]
        )
        assert idx == 2
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[2].source == "target = 1"
        assert len(nb.cells[2].outputs) == 1
        assert len(nb.cells[1].outputs) == 0  # the impostor code cell stays clean

    def test_update_by_id_missing_cell(self, nb_manager, nb_path):
        nb_manager.add_code_cell(nb_path, "x = 1")
        with pytest.raises(KeyError):
            nb_manager.update_cell_outputs_by_id(
                nb_path, "no_such_id", [{"type": "text", "content": "x"}]
            )

    def test_update_by_id_rejects_markdown(self, nb_manager, nb_path):
        nb_manager.add_markdown_cell(nb_path, "# md")
        cell_id = nb_manager.get_cell_id(nb_path, 0)
        with pytest.raises(ValueError):
            nb_manager.update_cell_outputs_by_id(
                nb_path, cell_id, [{"type": "text", "content": "x"}]
            )


class TestOutputConversionFidelity:
    def test_stderr_stream_name_preserved(self, nb_manager, nb_path):
        nb_manager.add_code_cell(
            nb_path,
            "warn",
            outputs=[{"type": "text", "content": "warning!\n", "name": "stderr"}],
        )
        nb = nb_manager.get_notebook(nb_path)
        out = nb.cells[0].outputs[0]
        assert out["output_type"] == "stream"
        assert out["name"] == "stderr"

    def test_default_stream_is_stdout(self, nb_manager, nb_path):
        nb_manager.add_code_cell(
            nb_path, "p", outputs=[{"type": "text", "content": "hi\n"}]
        )
        out = nb_manager.get_notebook(nb_path).cells[0].outputs[0]
        assert out["output_type"] == "stream"
        assert out["name"] == "stdout"

    def test_execute_result_semantics(self, nb_manager, nb_path):
        nb_manager.add_code_cell(
            nb_path,
            "1 + 1",
            outputs=[{"type": "text", "content": "2", "result": True}],
        )
        nb = nb_manager.get_notebook(nb_path)
        out = nb.cells[0].outputs[0]
        assert out["output_type"] == "execute_result"
        assert out["data"]["text/plain"] == "2"
        nbformat.validate(nb)
