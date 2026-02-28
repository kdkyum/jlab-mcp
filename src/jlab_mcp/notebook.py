import re
import shutil
import uuid
from pathlib import Path
from typing import Callable

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
        """Create an empty .ipynb notebook. Returns the path."""
        name = _sanitize_filename(name)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        nb_path = directory / f"{name}.ipynb"

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
        """Replace cell content and outputs. Returns the resolved cell index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        nb.cells[cell_index].source = new_code
        if outputs is not None:
            nb.cells[cell_index].outputs = self._convert_outputs(outputs)
        self.save_notebook(nb_path, nb)
        return cell_index

    def get_cell_source(self, nb_path: Path | str, cell_index: int) -> str:
        """Read cell source by index. Supports negative indexing."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        return nb.cells[cell_index].source

    def update_cell_outputs(
        self, nb_path: Path | str, cell_index: int, outputs: list[dict]
    ) -> int:
        """Update only the outputs of a cell (source unchanged). Returns resolved index."""
        nb = self.get_notebook(nb_path)
        cell_index = self._resolve_cell_index(nb, cell_index)
        nb.cells[cell_index].outputs = self._convert_outputs(outputs)
        self.save_notebook(nb_path, nb)
        return cell_index

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
        """Write notebook to disk after cleaning."""
        self.clean_notebook(notebook)
        nbformat.write(notebook, str(nb_path))

    def clean_notebook(self, notebook: nbformat.NotebookNode) -> None:
        """Ensure all cells have valid IDs (required by nbformat v5+)."""
        for cell in notebook.cells:
            if "id" not in cell or not cell["id"]:
                cell["id"] = uuid.uuid4().hex[:8]

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

    def restore_notebook(
        self,
        nb_path: Path | str,
        kernel_execute_fn: Callable[[str], list[dict]],
    ) -> list[str]:
        """Re-execute all code cells to restore kernel state.

        Returns list of error messages (empty if all succeeded).
        """
        nb = self.get_notebook(nb_path)
        errors: list[str] = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            outputs = kernel_execute_fn(cell.source)
            cell.outputs = self._convert_outputs(outputs)
            for out in outputs:
                if out.get("type") == "error":
                    errors.append(
                        f"Cell {i}: {out.get('ename', 'Error')}: "
                        f"{out.get('evalue', '')}"
                    )
        self.save_notebook(nb_path, nb)
        return errors

    def _convert_outputs(self, outputs: list[dict]) -> list:
        """Convert our output dicts to nbformat output objects."""
        nb_outputs = []
        for out in outputs:
            if out["type"] == "text":
                nb_outputs.append(
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
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
