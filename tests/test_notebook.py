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


class TestAddMarkdownCell:
    def test_add_markdown(self, nb_manager, nb_path):
        idx = nb_manager.add_markdown_cell(nb_path, "# Title\nSome text")
        assert idx == 0
        nb = nb_manager.get_notebook(nb_path)
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[0].source == "# Title\nSome text"


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
