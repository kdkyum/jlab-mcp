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
        self, nb_path: Path | str, code: str, outputs: list[dict] | None = None
    ) -> int:
        """Append a code cell. Returns the cell index."""
        nb = self.get_notebook(nb_path)
        cell = new_code_cell(source=code)
        if outputs:
            cell.outputs = self._convert_outputs(outputs)
        nb.cells.append(cell)
        self.save_notebook(nb_path, nb)
        return len(nb.cells) - 1

    def add_markdown_cell(self, nb_path: Path | str, markdown: str) -> int:
        """Append a markdown cell. Returns the cell index."""
        nb = self.get_notebook(nb_path)
        cell = new_markdown_cell(source=markdown)
        nb.cells.append(cell)
        self.save_notebook(nb_path, nb)
        return len(nb.cells) - 1

    def edit_cell(
        self,
        nb_path: Path | str,
        cell_index: int,
        new_code: str,
        outputs: list[dict] | None = None,
    ) -> int:
        """Replace cell content and outputs. Returns the resolved cell index."""
        nb = self.get_notebook(nb_path)
        # Support negative indexing
        if cell_index < 0:
            cell_index = len(nb.cells) + cell_index
        if cell_index < 0 or cell_index >= len(nb.cells):
            raise IndexError(
                f"Cell index {cell_index} out of range "
                f"(notebook has {len(nb.cells)} cells)"
            )
        nb.cells[cell_index].source = new_code
        if outputs is not None:
            nb.cells[cell_index].outputs = self._convert_outputs(outputs)
        self.save_notebook(nb_path, nb)
        return cell_index

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
