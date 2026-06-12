import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace path separators and other unsafe chars with underscores
    name = re.sub(r'[/\\:*?"<>|\x00]', "_", name)
    # Remove leading/trailing dots and spaces
    name = name.strip(". ")
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Truncate to reasonable length
    name = name[:200]
    if not name:
        name = "unnamed"
    return name


class NotebookManager:
    """Manages notebook state using nbformat."""

    @staticmethod
    def _resolve_cell_index(nb: nbformat.NotebookNode, cell_index: int) -> int:
        """Resolve a cell index (with negative indexing) and bounds-check."""
        if cell_index < 0:
            cell_index = len(nb.cells) + cell_index
        if cell_index < 0 or cell_index >= len(nb.cells):
            raise IndexError(
                f"Cell index {cell_index} out of range "
                f"(notebook has {len(nb.cells)} cells)"
            )
        return cell_index

    @staticmethod
    def _insert_cell(nb: nbformat.NotebookNode, cell, index: int) -> int:
        """Insert a cell at index. -1 = append, other negatives raise IndexError."""
        if index == -1:
            nb.cells.append(cell)
            return len(nb.cells) - 1
        if index < 0 or index > len(nb.cells):
            raise IndexError(
                f"Insert index {index} out of range "
                f"(notebook has {len(nb.cells)} cells)"
            )
        nb.cells.insert(index, cell)
        return index

    def create_notebook(self, name: str, directory: str | Path) -> Path:
        """Create an empty .ipynb notebook. Returns the path.

        Never overwrites: if {name}.ipynb already exists, appends _2, _3, ...
        (same collision handling as copy_notebook).
        """
        name = _sanitize_filename(name)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        nb_path = directory / f"{name}.ipynb"
        counter = 2
        while nb_path.exists():
            nb_path = directory / f"{name}_{counter}.ipynb"
            counter += 1

        nb = new_notebook()
        nb.metadata.update(
            {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                },
            }
        )
        self.save_notebook(nb_path, nb)
        return nb_path

    def add_code_cell(
        self,
        nb_path: Path | str,
        code: str,
        outputs: list[dict] | None = None,
        index: int = -1,
    ) -> int:
        """Add a code cell. Returns the cell index.

        Args:
            index: Position to insert. -1 = append (default).
                   0..len = insert at position. Other negatives raise IndexError.
        """
        nb = self.get_notebook(nb_path)
        cell = new_code_cell(source=code)
        if outputs:
            cell.outputs = self._convert_outputs(outputs)
        idx = self._insert_cell(nb, cell, index)
        self.save_notebook(nb_path, nb)
        return idx

    def add_markdown_cell(
        self, nb_path: Path | str, markdown: str, index: int = -1
    ) -> int:
        """Add a markdown cell. Returns the cell index.

        Args:
            index: Position to insert. -1 = append (default).
                   0..len = insert at position. Other negatives raise IndexError.
        """
        nb = self.get_notebook(nb_path)
        cell = new_markdown_cell(source=markdown)
        idx = self._insert_cell(nb, cell, index)
        self.save_notebook(nb_path, nb)
        return idx

    def edit_cell(
        self,
        nb_path: Path | str,
        cell_index: int,
        new_code: str,
        outputs: list[dict] | None = None,
    ) -> int:
        """Replace a code cell's content and outputs. Returns the resolved cell index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        cell = nb.cells[cell_index]
        if cell.cell_type != "code":
            # 'outputs' is only valid on code cells; writing it to a markdown
            # cell produces a schema-invalid notebook
            raise ValueError(
                f"Cell {cell_index} is a {cell.cell_type} cell, not a code "
                f"cell (use edit_markdown for markdown cells)"
            )
        cell.source = new_code
        if outputs is not None:
            cell.outputs = self._convert_outputs(outputs)
        self.save_notebook(nb_path, nb)
        return cell_index

    def get_cell_source(self, nb_path: Path | str, cell_index: int) -> str:
        """Read cell source by index. Supports negative indexing."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        return nb.cells[cell_index].source

    def get_cell_type(self, nb_path: Path | str, cell_index: int) -> str:
        """Read cell type by index. Supports negative indexing."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        return nb.cells[cell_index].cell_type

    def update_cell_outputs(
        self, nb_path: Path | str, cell_index: int, outputs: list[dict]
    ) -> int:
        """Update only the outputs of a cell (source unchanged). Returns resolved index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        if nb.cells[cell_index].cell_type != "code":
            raise ValueError(
                f"Cell {cell_index} is a {nb.cells[cell_index].cell_type} "
                f"cell; only code cells have outputs"
            )
        nb.cells[cell_index].outputs = self._convert_outputs(outputs)
        self.save_notebook(nb_path, nb)
        return cell_index

    def get_cell_id(self, nb_path: Path | str, cell_index: int) -> str:
        """Read a cell's nbformat id by index. Supports negative indexing.

        Cell ids are stable across inserts/deletes of other cells, unlike
        positional indices — use them to refer to a cell across a long
        kernel execution.
        """
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        return nb.cells[cell_index]["id"]

    def update_cell_outputs_by_id(
        self, nb_path: Path | str, cell_id: str, outputs: list[dict]
    ) -> int:
        """Update the outputs of the cell with the given nbformat id.

        Returns the cell's current index. Raises KeyError if no cell with
        that id exists (e.g. it was deleted while the code was executing).
        """
        nb = self.get_notebook(nb_path)
        for i, cell in enumerate(nb.cells):
            if cell.get("id") == cell_id:
                if cell.cell_type != "code":
                    raise ValueError(
                        f"Cell {cell_id} is a {cell.cell_type} cell; "
                        f"only code cells have outputs"
                    )
                cell.outputs = self._convert_outputs(outputs)
                self.save_notebook(nb_path, nb)
                return i
        raise KeyError(f"No cell with id {cell_id!r} in {nb_path}")

    def get_code_cells(self, nb_path: Path | str) -> list[dict]:
        """Return info for all code cells.

        Returns:
            List of dicts: [{"index": i, "source": ...}, ...]
            The index is the absolute notebook cell index (including markdown cells).
        """
        nb = self.get_notebook(nb_path)
        return [
            {"index": i, "source": cell.source}
            for i, cell in enumerate(nb.cells)
            if cell.cell_type == "code"
        ]

    def get_notebook(self, nb_path: Path | str) -> nbformat.NotebookNode:
        """Read a notebook from disk."""
        return nbformat.read(str(nb_path), as_version=4)

    def save_notebook(
        self, nb_path: Path | str, notebook: nbformat.NotebookNode
    ) -> None:
        """Write notebook to disk atomically after cleaning."""
        self.clean_notebook(notebook)
        nb_path = Path(nb_path)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(nb_path.parent), suffix=".ipynb.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                nbformat.write(notebook, f)
            os.replace(tmp_path, str(nb_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def clean_notebook(self, notebook: nbformat.NotebookNode) -> None:
        """Ensure all cells have valid IDs (required by nbformat v5+)."""
        for cell in notebook.cells:
            if "id" not in cell or not cell["id"]:
                cell["id"] = uuid.uuid4().hex[:8]

    def delete_cell(self, nb_path: Path | str, cell_index: int) -> int:
        """Delete a cell by index. Returns the resolved index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        del nb.cells[cell_index]
        self.save_notebook(nb_path, nb)
        return cell_index

    def edit_markdown_cell(
        self, nb_path: Path | str, cell_index: int, markdown: str
    ) -> int:
        """Replace a markdown cell's content. Returns the resolved index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        if nb.cells[cell_index].cell_type != "markdown":
            raise ValueError(
                f"Cell {cell_index} is a {nb.cells[cell_index].cell_type} cell, "
                f"not a markdown cell"
            )
        nb.cells[cell_index].source = markdown
        self.save_notebook(nb_path, nb)
        return cell_index

    def get_cell_count(self, nb_path: Path | str) -> int:
        """Get number of cells in a notebook."""
        nb = self.get_notebook(nb_path)
        return len(nb.cells)

    def copy_notebook(self, src_path: Path | str, suffix: str = "_continued") -> Path:
        """Copy a notebook with a suffix. Returns the new path.

        If the destination already exists, appends _2, _3, etc.
        """
        src = Path(src_path)
        dst = src.with_stem(src.stem + suffix)
        if dst.exists():
            counter = 2
            while dst.exists():
                dst = src.with_stem(f"{src.stem}{suffix}_{counter}")
                counter += 1
        shutil.copy2(src, dst)
        return dst

    def _convert_outputs(self, outputs: list[dict]) -> list:
        """Convert our output dicts to nbformat output objects."""
        nb_outputs = []
        for out in outputs:
            if out["type"] == "text":
                if out.get("result"):
                    # Expression value (Out[n]) — keep execute_result semantics
                    nb_outputs.append(
                        nbformat.v4.new_output(
                            output_type="execute_result",
                            data={"text/plain": out["content"]},
                            execution_count=None,
                        )
                    )
                else:
                    nb_outputs.append(
                        nbformat.v4.new_output(
                            output_type="stream",
                            name=out.get("name", "stdout"),
                            text=out["content"],
                        )
                    )
            elif out["type"] == "image":
                nb_outputs.append(
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={"image/png": out["content"]},
                    )
                )
            elif out["type"] == "error":
                nb_outputs.append(
                    nbformat.v4.new_output(
                        output_type="error",
                        ename=out.get("ename", "Error"),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", []),
                    )
                )
        return nb_outputs
