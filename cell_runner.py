#!/usr/bin/env python3
import csv
import curses
import logging
import sys
from enum import Enum
from typing import Any, List, Set, Tuple, TypeAlias, Union
from decimal import Decimal

CursesWindow: TypeAlias = Any  # Represents a curses window object

CELL_WIDTH: int = 16
CELL_REF_SEPARATOR = " :: "


class Keys(Enum):
    ENTER = 10
    ESC = 27


class OperationMode(Enum):
    NORMAL = 1
    INSERT = 2
    COMMAND = 4
    VISUAL = 8

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class CellType(Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    DECIMAL = "decimal"
    EMPTY = "empty"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Function to set up a logger with file and console handlers"""

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler(log_file)  # File handler

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


CURRENT_LOGGER: logging.Logger = setup_logger("CELL RUNNER", "log.txt")


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Number of times this word appears
        self.value = None  # The actual value stored


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, value: str):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = value
        node.count += 1

    def find_suggestions(self, prefix: str) -> List[Tuple[str, str, int]]:
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = []
        self._collect_suggestions(node, prefix, suggestions)
        # Sort by count (frequency) and return all matches
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions

    def _collect_suggestions(self, node: TrieNode, prefix: str, suggestions: List[Tuple[str, str, int]]):
        if node.is_end:
            suggestions.append((prefix, node.value, node.count))
        
        for char, child in node.children.items():
            self._collect_suggestions(child, prefix + char, suggestions)


class Format:
    def __init__(self, format_type: str, *args):
        self.format_type = format_type
        self.args = args

    def apply(self, value: Any) -> str:
        if self.format_type == "s":  # Separator format
            # Only apply separator format to numeric types (INT, FLOAT, DECIMAL)
            if isinstance(value, bool):
                return str(value).lower()
            if not isinstance(value, (int, float, Decimal)):
                return str(value)
            
            # Convert to string with proper decimal places
            if isinstance(value, float):
                str_value = f"{value:.2f}"
            else:
                str_value = str(value)
            
            # Split into integer and decimal parts
            if '.' in str_value:
                int_part, dec_part = str_value.split('.')
            else:
                int_part, dec_part = str_value, ""
            
            # Add thousands separator if specified
            if self.args and self.args[0]:
                separator = self.args[0]
                # Reverse the integer part for easier grouping
                int_part = int_part[::-1]
                # Group by 3 and join with separator
                groups = [int_part[i:i+3] for i in range(0, len(int_part), 3)]
                int_part = separator.join(groups)
                # Reverse back
                int_part = int_part[::-1]
            
            # Combine parts
            return f"{int_part}{'.' + dec_part if dec_part else ''}"
        return str(value)  # Default formatting


class Cell:
    def __init__(self, content: str = ""):
        self.content = content
        self.cursor_pos = 0
        self.formula = None
        self.value: Union[int, float, str, bool, Decimal, None] = None
        self.cell_type = CellType.EMPTY
        self.dependencies: Set[Tuple[int, int]] = set()  # Set of (row, col) tuples
        self.dependents: Set[Tuple[int, int]] = set()  # Cells that depend on this cell
        self.editing_value = None  # Temporary value while editing
        self.format = None  # Cell format
        if content:
            self.set_content(content)

    def set_content(self, content: str):
        self.content = content
        if content.startswith("="):
            self.formula = content[1:]
            self.dependencies = set()
            self.value = None
            self.cell_type = CellType.EMPTY
        else:
            self.formula = None
            if not content:
                self.value = None
                self.cell_type = CellType.EMPTY
            else:
                # Try to determine the type and convert the value
                try:
                    # Try bool first (case-insensitive)
                    if content.lower() in ('true', 'false'):
                        self.value = content.lower() == 'true'
                        self.cell_type = CellType.BOOL
                    # Try int
                    elif content.isdigit() or (content.startswith('-') and content[1:].isdigit()):
                        self.value = int(content)
                        self.cell_type = CellType.INT
                    # Try decimal (numbers with decimal point)
                    elif '.' in content and all(c.isdigit() or c == '.' or c == '-' for c in content):
                        self.value = Decimal(content)
                        self.cell_type = CellType.DECIMAL
                    # Try float
                    else:
                        self.value = float(content)
                        self.cell_type = CellType.FLOAT
                except (ValueError, TypeError):
                    # If all else fails, treat as string
                    self.value = content
                    self.cell_type = CellType.STRING
        # Clear editing_value to force display update
        self.editing_value = None

    def get_display_value(self, column_format=None, row_format=None) -> str:
        if self.editing_value is not None:
            return self.editing_value
        if self.formula is not None:
            return f"={self.formula}"
        if self.value is None:
            return ""
        
        # Apply format in order: cell format > column format > row format
        if self.format:
            return self.format.apply(self.value)
        elif column_format:
            return column_format.apply(self.value)
        elif row_format:
            return row_format.apply(self.value)
        
        # Default formatting
        if self.cell_type == CellType.BOOL:
            return str(self.value).lower()
        elif self.cell_type == CellType.INT:
            return str(self.value)
        elif self.cell_type == CellType.FLOAT:
            return f"{self.value:.2f}"
        elif self.cell_type == CellType.DECIMAL:
            return str(self.value)
        else:  # STRING or EMPTY
            return str(self.value)

    def start_editing(self):
        # Store the current display value as editing value to preserve format
        if self.value is not None:
            self.editing_value = self.get_display_value()
        else:
            self.editing_value = self.content

    def finish_editing(self):
        if self.editing_value is not None:
            # Store the current display value before setting content
            temp_value = self.editing_value
            self.editing_value = None
            self.set_content(temp_value)
            # Force display update with current formats
            if self.value is not None:
                self.editing_value = self.get_display_value()

    def set_format(self, format_type: str, *args):
        # Preserve existing format if it exists
        if self.format:
            old_format = self.format
            # Only update if the new format is different
            if old_format.format_type != format_type or old_format.args != args:
                self.format = Format(format_type, *args)
                # Force display update for the current cell
                if self.value is not None:
                    self.editing_value = self.get_display_value()
        else:
            self.format = Format(format_type, *args)
            # Force display update for the current cell
            if self.value is not None:
                self.editing_value = self.get_display_value()

    def clear_format(self):
        self.format = None
        # Force display update for the current cell
        if self.value is not None:
            self.editing_value = self.get_display_value()


class Column:
    def __init__(self, index: int):
        self.index = index
        self.width = max(2, CELL_WIDTH)  # Ensure minimum width of 2
        self.visible = True
        self.frozen = False
        self.format = None  # Column format
        self.type = None  # For future column type hints

    def get_letter(self) -> str:
        # Convert index to base-26 (A=0, B=1, ..., Z=25)
        result = []
        n = self.index
        while True:
            n, remainder = divmod(n, 26)
            result.append(chr(ord('A') + remainder))
            if n == 0:
                break
        return ''.join(reversed(result))

    def get_display_width(self) -> int:
        return self.width

    def set_width(self, width: int):
        self.width = max(2, width)  # Ensure width is at least 2

    def toggle_visibility(self):
        self.visible = not self.visible

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def set_format(self, format_type: str, *args):
        # Preserve existing format if it exists
        if self.format:
            old_format = self.format
            # Only update if the new format is different
            if old_format.format_type != format_type or old_format.args != args:
                self.format = Format(format_type, *args)
        else:
            self.format = Format(format_type, *args)

    def clear_format(self):
        self.format = None


class Row:
    def __init__(self, index: int):
        self.index = index
        self.height = 2  # Height in display lines (1 for content + 1 for border)
        self.visible = True
        self.frozen = False
        self.format = None  # Row format
        self.type = None  # For future row type hints
        self.cells: List[Cell] = []

    def get_number(self) -> str:
        return str(self.index + 1)

    def get_display_height(self) -> int:
        return self.height

    def set_height(self, height: int):
        self.height = max(2, height)  # Ensure height is at least 2 (1 for content + 1 for border)

    def toggle_visibility(self):
        self.visible = not self.visible

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def set_format(self, format_type: str, *args):
        # Preserve existing format if it exists
        if self.format:
            old_format = self.format
            # Only update if the new format is different
            if old_format.format_type != format_type or old_format.args != args:
                self.format = Format(format_type, *args)
        else:
            self.format = Format(format_type, *args)

    def clear_format(self):
        self.format = None

    def add_cell(self, cell: Cell):
        self.cells.append(cell)

    def get_cell(self, col_index: int) -> Cell:
        while len(self.cells) <= col_index:
            self.cells.append(Cell())
        return self.cells[col_index]

    def delete_cell(self, col_index: int):
        if 0 <= col_index < len(self.cells):
            self.cells.pop(col_index)


class CellRunner:
    def __init__(self):
        self.reset()

    def enter_insert_mode(self, initial_value: str = ""):
        """Helper method to enter insert mode with optional initial value"""
        self.mode = OperationMode.INSERT
        self.current_input = initial_value
        self.input_cursor_pos = len(initial_value)
        self.get_current_cell().start_editing()
        self.command_multiplier = 0
        self.suggestions = []
        self.selected_suggestion = -1
        self.suggestion_scroll_offset = 0

    def reset(self):
        # Create a reasonable number of columns initially (e.g., 10)
        self.columns: List[Column] = [Column(i) for i in range(10)]
        self.rows: List[Row] = [Row(0)]
        self.current_row: int = 0
        self.current_col: int = 0
        self.mode = OperationMode.NORMAL
        self.command_buffer = ""
        self.command_multiplier: int = 0
        self.current_input = ""
        self.input_cursor_pos = 0
        self.current_file: str = ""
        self.trie = None  # Will be initialized when indexing is requested
        self.suggestions = []  # Current suggestions for autocomplete
        self.selected_suggestion = -1  # Index of currently selected suggestion (-1 means no selection)
        self.suggestion_scroll_offset = 0  # Offset for scrolling through suggestions
        self.visible_suggestions = 5  # Number of suggestions to show at once
        self.indexed_columns: Set[int] = set()  # Track which columns are indexed
        self.indexed_rows: Set[int] = set()  # Track which rows are indexed
        self.error_message = ""  # For displaying command errors
        # Visual mode state
        self.visual_start_row: int = 0
        self.visual_start_col: int = 0
        self.visual_end_row: int = 0
        self.visual_end_col: int = 0

        # Initialize cells for the first row
        for col in range(len(self.columns)):
            self.rows[0].add_cell(Cell())

    def get_current_cell(self) -> Cell:
        return self.rows[self.current_row].get_cell(self.current_col)

    def get_cell_reference(self, row: int, col: int) -> str:
        self.ensure_grid_size(row, col)
        return f"{self.columns[col].get_letter()}{row + 1}"

    def ensure_grid_size(self, row: int, col: int):
        # Ensure enough rows
        while len(self.rows) <= row:
            self.rows.append(Row(len(self.rows)))

        # Ensure enough columns
        while len(self.columns) <= col:
            self.columns.append(Column(len(self.columns)))

        # Ensure each row has enough cells
        for r in self.rows:
            while len(r.cells) <= col:
                r.add_cell(Cell())

    def add_cell_right(self):
        self.ensure_grid_size(self.current_row, self.current_col + 1)
        self.current_col += 1

    def add_cell_below(self):
        self.ensure_grid_size(self.current_row + 1, self.current_col)
        self.current_row += 1

    def add_cell_above(self):
        new_row = Row(self.current_row)
        for col in range(len(self.columns)):
            new_row.add_cell(Cell())
        self.rows.insert(self.current_row, new_row)
        # Update row indices
        for i, row in enumerate(self.rows):
            row.index = i

    def clear_cell(self):
        self.get_current_cell().set_content("")

    def delete_cell(self):
        if len(self.rows) > 1 or len(self.columns) > 1:
            # Remove the current cell from the row
            self.rows[self.current_row].delete_cell(self.current_col)

            # If the row is empty, remove it
            if not self.rows[self.current_row].cells:
                self.rows.pop(self.current_row)
                # Update row indices
                for i, row in enumerate(self.rows):
                    row.index = i

            # Adjust current position
            self.current_col = max(0, self.current_col - 1)
            self.current_row = min(self.current_row, len(self.rows) - 1)

    def find_last_non_empty_in_row(self, row: int) -> int:
        for col in range(len(self.rows[row].cells) - 1, -1, -1):
            if self.rows[row].cells[col].content:
                return col
        return -1

    def delete_current_row(self):
        if len(self.rows) > 1:
            self.rows.pop(self.current_row)
            # Update row indices
            for i, row in enumerate(self.rows):
                row.index = i
            self.current_row = min(self.current_row, len(self.rows) - 1)

    def update_cell_values(self):
        return

    def move_to_row(self, new_row: int):
        self.ensure_grid_size(new_row, self.current_col)
        self.current_row = new_row - 1

    def move_to_cell(self, new_cell: int):
        self.ensure_grid_size(self.current_row, new_cell)
        self.current_cell = new_cell

    def index_table(self):
        """Index all string values in the table"""
        if not self.trie:
            self.trie = Trie()
        for row in self.rows:
            for cell in row.cells:
                if cell.cell_type == CellType.STRING and cell.value:
                    self.trie.insert(cell.value, cell.value)
        self.indexed_columns = set(range(len(self.columns)))
        self.indexed_rows = set(range(len(self.rows)))

    def index_current_column(self):
        """Index string values in the current column"""
        if not self.trie:
            self.trie = Trie()
        for row in self.rows:
            cell = row.get_cell(self.current_col)
            if cell.cell_type == CellType.STRING and cell.value:
                self.trie.insert(cell.value, cell.value)
        self.indexed_columns.add(self.current_col)

    def index_current_row(self):
        """Index string values in the current row"""
        if not self.trie:
            self.trie = Trie()
        for cell in self.rows[self.current_row].cells:
            if cell.cell_type == CellType.STRING and cell.value:
                self.trie.insert(cell.value, cell.value)
        self.indexed_rows.add(self.current_row)

    def clear_table_index(self):
        """Clear the entire index"""
        self.trie = None
        self.indexed_columns.clear()
        self.indexed_rows.clear()

    def clear_column_index(self):
        """Clear index for the current column"""
        if not self.trie:
            return
        # Rebuild the index excluding the current column
        new_trie = Trie()
        for row_idx, row in enumerate(self.rows):
            for col_idx, cell in enumerate(row.cells):
                # Only include cells from indexed columns and rows
                if (col_idx != self.current_col and 
                    col_idx in self.indexed_columns and 
                    row_idx in self.indexed_rows and 
                    cell.cell_type == CellType.STRING and 
                    cell.value):
                    new_trie.insert(cell.value, cell.value)
        self.trie = new_trie
        self.indexed_columns.discard(self.current_col)

    def clear_row_index(self):
        """Clear index for the current row"""
        if not self.trie:
            return
        # Rebuild the index excluding the current row
        new_trie = Trie()
        for row_idx, row in enumerate(self.rows):
            if row_idx != self.current_row:
                for col_idx, cell in enumerate(row.cells):
                    # Only include cells from indexed columns and rows
                    if (col_idx in self.indexed_columns and 
                        row_idx in self.indexed_rows and 
                        cell.cell_type == CellType.STRING and 
                        cell.value):
                        new_trie.insert(cell.value, cell.value)
        self.trie = new_trie
        self.indexed_rows.discard(self.current_row)

    def get_suggestions(self, prefix: str) -> List[Tuple[str, str, int]]:
        """Get suggestions for the given prefix"""
        if not self.trie or not prefix:
            return []
        return self.trie.find_suggestions(prefix)

    def get_current_suggestion(self) -> Tuple[str, str, int]:
        """Get the currently selected suggestion"""
        if 0 <= self.selected_suggestion < len(self.suggestions):
            return self.suggestions[self.selected_suggestion]
        return None

    def get_visible_suggestions(self) -> List[Tuple[str, str, int]]:
        """Get the currently visible batch of suggestions"""
        if not self.suggestions:
            return []
        start_idx = self.suggestion_scroll_offset
        end_idx = min(start_idx + self.visible_suggestions, len(self.suggestions))
        return self.suggestions[start_idx:end_idx]

    def scroll_suggestions(self, direction: int):
        """Scroll through suggestions in batches"""
        if not self.suggestions:
            return
            
        # Calculate new scroll offset
        new_offset = self.suggestion_scroll_offset + (direction * self.visible_suggestions)
        
        # Ensure we don't scroll past the beginning or end
        if new_offset < 0:
            new_offset = 0
        elif new_offset >= len(self.suggestions):
            new_offset = max(0, len(self.suggestions) - self.visible_suggestions)
            
        self.suggestion_scroll_offset = new_offset
        
        # Adjust selected suggestion to be within visible range
        if self.selected_suggestion >= 0:
            visible_start = self.suggestion_scroll_offset
            visible_end = visible_start + self.visible_suggestions
            if self.selected_suggestion < visible_start:
                self.selected_suggestion = visible_start
            elif self.selected_suggestion >= visible_end:
                self.selected_suggestion = visible_end - 1

    def apply_suggestion(self):
        """Apply the currently selected suggestion"""
        if self.get_current_suggestion():
            prefix, value, _ = self.get_current_suggestion()
            self.current_input = value
            self.input_cursor_pos = len(value)
            self.get_current_cell().editing_value = value
            self.suggestions = []
            self.selected_suggestion = -1
            self.suggestion_scroll_offset = 0


def normal_write(runner: CellRunner, destination: str):
    # TODO add ranges and different formats

    with open(destination, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        content_grid = [[cell.content for cell in row.cells] for row in runner.rows]
        writer.writerows(content_grid)
    runner.current_file = destination


def normal_free(runner: CellRunner):
    runner.reset()
    runner.current_file = ""


def normal_read(runner: CellRunner, source: str):
    with open(source, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)  # Create a CSV reader object
        runner.get_current_cell().set_content(runner.current_input)

        new_grid: list[list[Cell]] = []
        # Iterate over rows in the file
        for row in reader:
            new_row: list[Cell] = []
            for col in row:
                new_element = Cell()
                new_element.set_content(col)
                new_row.append(new_element)
            new_grid.append(new_row)

        runner.rows = [Row(i) for i in range(len(new_grid))]
        for i, row in enumerate(new_grid):
            for j, cell in enumerate(row):
                runner.rows[i].add_cell(cell)

    runner.current_file = source


def display_grid(stdscr: CursesWindow, runner: CellRunner) -> None:
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    height = int(height)
    width = int(width)

    # Calculate total width of visible columns
    visible_width = 0
    visible_columns = []
    for col in runner.columns:
        if visible_width + col.get_display_width() <= width - 2:  # -2 for borders
            visible_width += col.get_display_width()
            visible_columns.append(col)
        else:
            break

    # Calculate horizontal scroll position based on current column
    horizontal_start = max(0, runner.current_col - len(visible_columns) // 2)
    horizontal_end = min(len(runner.columns), horizontal_start + len(visible_columns))

    # Adjust if we're near the end
    if horizontal_end > len(runner.columns):
        horizontal_end = len(runner.columns)
        horizontal_start = max(0, horizontal_end - len(visible_columns))
    elif horizontal_end == len(runner.columns) - 1:
        # If we're at the last column, make sure it's visible
        horizontal_start = max(0, horizontal_end - len(visible_columns))
        horizontal_end = len(runner.columns)

    # Calculate total height of visible rows
    visible_height = 0
    visible_rows = []
    for row in runner.rows:
        if visible_height + row.get_display_height() <= height - 2:  # -2 for status bar and ruler
            visible_height += row.get_display_height()
            visible_rows.append(row)
        else:
            break

    vertical_start: int = runner.current_row - len(visible_rows) // 2
    vertical_end: int = runner.current_row + len(visible_rows) // 2

    # Adjust vertical scrolling to ensure last row is visible
    if vertical_end >= len(runner.rows):
        vertical_end = len(runner.rows)
        vertical_start = max(0, vertical_end - len(visible_rows))
    elif vertical_start < 0:
        vertical_start = 0
        vertical_end = min(len(runner.rows), len(visible_rows))

    # Update cell values before display
    runner.update_cell_values()

    border = 1 + len(str(vertical_end))

    # Draw horizontal ruler (column letters) with a border line
    ruler_y = 0
    current_x = border
    for col in runner.columns[horizontal_start:horizontal_end]:
        col_letter = col.get_letter()
        col_width = col.get_display_width()
        if current_x + col_width <= width - 1:
            if runner.current_col == col.index:
                color_pair = 4
            else:
                color_pair = 1

            # Calculate center position for the letter
            letter_x = current_x + (col_width - len(col_letter)) // 2
            if letter_x < width - 1:
                stdscr.addstr(ruler_y, letter_x, col_letter, curses.color_pair(color_pair))
            current_x += col_width

    # Draw a border line below the header
    stdscr.addstr(ruler_y + 1, 0, "─" * (width - 1))

    # Draw vertical ruler (row numbers) and cells
    current_y = 2  # Start after the ruler and border line
    for row in runner.rows[vertical_start:vertical_end]:
        row_num = row.get_number()
        row_height = row.get_display_height()
        
        if current_y + row_height > height - 1:
            break

        # Draw row number
        if runner.current_row == row.index:
            stdscr.addstr(current_y, 0, row_num.rjust(border), curses.color_pair(4))
        else:
            stdscr.addstr(current_y, 0, row_num.rjust(border), curses.color_pair(1))

        # Draw cell borders and content
        current_x = border
        for col in runner.columns[horizontal_start:horizontal_end]:
            col_width = col.get_display_width()
            if current_x + col_width > width - 1:
                break

            # Draw vertical border
            stdscr.addstr(current_y, current_x, "│")
            
            # Draw cell content
            cell = row.get_cell(col.index)
            content = cell.get_display_value(
                column_format=runner.columns[col.index].format,
                row_format=row.format
            )
            
            # Determine if cell is in visual selection
            is_visual_selected = False
            if runner.mode == OperationMode.VISUAL:
                min_row = min(runner.visual_start_row, runner.visual_end_row)
                max_row = max(runner.visual_start_row, runner.visual_end_row)
                min_col = min(runner.visual_start_col, runner.visual_end_col)
                max_col = max(runner.visual_start_col, runner.visual_end_col)
                is_visual_selected = (
                    min_row <= row.index <= max_row and
                    min_col <= col.index <= max_col
                )
            
            # Determine cell color
            if row.index == runner.current_row and col.index == runner.current_col:
                color_pair = 4  # Current cell
            elif is_visual_selected:
                color_pair = 3  # Visual selection
            else:
                color_pair = 1  # Normal (white)

            # Draw empty cell with underscore for current cell, spaces for others
            if not content:
                if row.index == runner.current_row and col.index == runner.current_col:
                    content = "_" * (col_width - 2)  # -2 for borders
                else:
                    content = " " * (col_width - 2)  # -2 for borders
            stdscr.addstr(current_y, current_x + 1, content[:col_width-2], curses.color_pair(color_pair))
            
            current_x += col_width

        # Draw horizontal border
        stdscr.addstr(current_y + row_height - 1, 0, "─" * (width - 1))
        current_y += row_height

    # Draw suggestions if in insert mode and we have suggestions
    if runner.mode == OperationMode.INSERT and runner.suggestions:
        suggestion_y = height - 3  # Start two lines above the status bar
        
        # Get visible suggestions
        visible_suggestions = runner.get_visible_suggestions()
        
        # Find the longest suggestion text
        max_suggestion_length = max(
            len(f"{prefix} ({count})") 
            for prefix, _, count in visible_suggestions
        )
        
        # Create a background line with the maximum length
        background_line = " " * max_suggestion_length
        
        # Add scroll indicators if there are more suggestions
        if runner.suggestion_scroll_offset > 0:
            stdscr.addstr(suggestion_y, 0, "▲", curses.color_pair(2))
        if runner.suggestion_scroll_offset + runner.visible_suggestions < len(runner.suggestions):
            stdscr.addstr(suggestion_y - len(visible_suggestions) - 1, 0, "▼", curses.color_pair(2))
        
        for i, (prefix, value, count) in enumerate(visible_suggestions):
            if suggestion_y - i < 0:  # Don't write above the screen
                break
                
            # First draw the background
            if len(background_line) > width - 1:
                background_line = background_line[:width - 1]
            stdscr.addstr(suggestion_y - i, 0, background_line)
            
            # Then draw the suggestion text
            suggestion_text = f"{prefix} ({count})"
            if len(suggestion_text) > width - 1:
                suggestion_text = suggestion_text[:width - 1]
                
            # Highlight selected suggestion
            if i + runner.suggestion_scroll_offset == runner.selected_suggestion:
                stdscr.addstr(suggestion_y - i, 0, suggestion_text, curses.color_pair(4))
            else:
                stdscr.addstr(suggestion_y - i, 0, suggestion_text, curses.color_pair(2))

    mode_str = f"Mode: {runner.mode}"
    if runner.command_buffer or runner.command_multiplier:
        mode_str += f" | Command: {runner.command_multiplier}{runner.command_buffer}"
    
    file_str = f" | File: {runner.current_file}" if runner.current_file else ""

    # Get cell reference and content for all modes
    cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
    cell = runner.get_current_cell()
    cell_type = cell.cell_type
    cell_value = cell.get_display_value()
    cell_info_prefix = f"{cell_ref} <{cell_type}>"

    # Calculate status bar position
    status_y = height - 1
    error_y = height - 2
    cell_info_y = height - 2

    # Calculate total rows and columns
    total_rows = len(runner.rows)
    total_cols = len(runner.columns)
    
    # Get column range using letters
    first_col = runner.columns[0].get_letter()
    last_col = runner.columns[-1].get_letter()
    grid_info = f" | Grid: {total_rows} rows, {first_col}-{last_col}"

    # Draw cell info on its own line
    if len(cell_info_prefix) > width - 1:
        cell_info_prefix = cell_info_prefix[:width - 1]
    stdscr.addstr(cell_info_y, 0, cell_info_prefix, curses.color_pair(1))

    if runner.mode == OperationMode.INSERT:
        # Show input in the value bar with cursor
        input_str = f"{runner.current_input}"
        if runner.input_cursor_pos <= len(input_str):
            input_str = (
                input_str[: runner.input_cursor_pos]
                + "█"
                + input_str[runner.input_cursor_pos :]
            )
        if len(input_str) > width - len(cell_info_prefix) - 1:
            input_str = input_str[:width - len(cell_info_prefix) - 1]
        stdscr.addstr(cell_info_y, len(cell_info_prefix), " " + input_str, curses.color_pair(2))
        
        # Show status bar without input
        final_input_str = f"{mode_str}{file_str}{grid_info}"
        if runner.error_message:
            # Show error message in red above the status bar
            stdscr.addstr(error_y, 0, f"{runner.error_message}", curses.color_pair(4))
            # Show status bar
            stdscr.addstr(status_y, 0, final_input_str, curses.color_pair(2))
        else:
            stdscr.addstr(status_y, 0, final_input_str, curses.color_pair(2))
    elif runner.mode == OperationMode.COMMAND:
        input_str = f"{runner.current_input}"
        if runner.input_cursor_pos <= len(input_str):
            input_str = (
                ":"
                + input_str[: runner.input_cursor_pos]
                + "█"
                + input_str[runner.input_cursor_pos :]
            )
        final_input_str = f"{mode_str}{file_str}{grid_info} | {input_str}"
        if runner.error_message:
            # Show error message in red above the status bar
            stdscr.addstr(error_y, 0, f"{runner.error_message}", curses.color_pair(4))
            # Show status bar
            stdscr.addstr(status_y, 0, final_input_str, curses.color_pair(2))
        else:
            stdscr.addstr(status_y, 0, final_input_str, curses.color_pair(2))
    else:
        # Show current value in normal mode
        if len(cell_value) > width - len(cell_info_prefix) - 1:
            cell_value = cell_value[:width - len(cell_info_prefix) - 1]
        stdscr.addstr(cell_info_y, len(cell_info_prefix), " " + cell_value, curses.color_pair(1))
        
        normal_str = f"{mode_str}{file_str}{grid_info}"
        if runner.error_message:
            # Show error message in red above the status bar
            stdscr.addstr(error_y, 0, f"{runner.error_message}", curses.color_pair(4))
            # Show status bar
            stdscr.addstr(status_y, 0, normal_str, curses.color_pair(1))
        else:
            stdscr.addstr(status_y, 0, normal_str, curses.color_pair(1))

    stdscr.refresh()


def handle_input(runner: CellRunner, key: int) -> bool:
    if runner.mode == OperationMode.NORMAL:
        return handle_normal_mode(runner, key)
    elif runner.mode == OperationMode.INSERT:
        return handle_insert_mode(runner, key)
    elif runner.mode == OperationMode.COMMAND:
        return handle_command_mode(runner, key)
    elif runner.mode == OperationMode.VISUAL:
        return handle_visual_mode(runner, key)
    return True


def handle_normal_mode(runner: CellRunner, key: int) -> bool:
    # Clear error message on any action in normal mode
    runner.error_message = ""
    
    if ord("0") <= key <= ord("9"):
        runner.command_multiplier = runner.command_multiplier * 10 + (key - ord("0"))
    elif key == ord("j"):
        mult = max(1, runner.command_multiplier)
        runner.current_row = min(len(runner.rows) - 1, runner.current_row + mult)
        runner.command_multiplier = 0
    elif key == ord("k"):
        mult = max(1, runner.command_multiplier)
        runner.current_row = max(0, runner.current_row - mult)
        runner.command_multiplier = 0
    elif key == ord("h"):
        mult = max(1, runner.command_multiplier)
        runner.current_col = max(0, runner.current_col - mult)
        runner.command_multiplier = 0
    elif key == ord("l"):
        mult = max(1, runner.command_multiplier)
        runner.current_col = min(
            len(runner.rows[0].cells) - 1,
            runner.current_col + mult,
        )
        runner.command_multiplier = 0
    elif key == ord("$"):  # Move to end of line
        last_col = runner.find_last_non_empty_in_row(runner.current_row)
        if last_col >= 0:
            runner.current_col = last_col
        else:
            runner.current_col = 0
        runner.command_multiplier = 0
    elif key == ord("^"):  # Move to start of line
        runner.current_col = 0
        runner.command_multiplier = 0
    elif key == ord("i"):
        runner.enter_insert_mode(runner.get_current_cell().content)
    elif key == ord("a"):
        mult = max(1, runner.command_multiplier)
        for _ in range(mult):
            runner.add_cell_right()
        runner.enter_insert_mode()
    elif key == ord("A"):
        last_col = runner.find_last_non_empty_in_row(runner.current_row)
        if last_col >= 0:
            runner.current_col = last_col + 1
            runner.ensure_grid_size(runner.current_row, runner.current_col)
        runner.enter_insert_mode()
    elif key == ord("o"):
        mult = max(1, runner.command_multiplier)
        for _ in range(mult):
            runner.add_cell_below()
        runner.enter_insert_mode()
    elif key == ord("O"):
        mult = max(1, runner.command_multiplier)
        for _ in range(mult):
            runner.add_cell_above()
        runner.enter_insert_mode()
    elif key == ord("x"):
        mult = max(1, runner.command_multiplier)
        for _ in range(mult):
            runner.clear_cell()
        runner.command_multiplier = 0
    elif key == ord("d"):
        if runner.command_buffer == "d":
            mult = max(1, runner.command_multiplier)
            for _ in range(mult):
                runner.delete_current_row()
            runner.command_buffer = ""
            runner.command_multiplier = 0
        else:
            runner.command_buffer = "d"
    elif key == ord(":"):
        runner.mode = OperationMode.COMMAND
        runner.command_multiplier = 0
    elif key == ord("v"):
        # Enter visual mode
        runner.mode = OperationMode.VISUAL
        runner.visual_start_row = runner.current_row
        runner.visual_start_col = runner.current_col
        runner.visual_end_row = runner.current_row
        runner.visual_end_col = runner.current_col
    elif key == Keys.ESC.value:
        runner.command_buffer = ""
        runner.command_multiplier = 0
    # Add Ctrl+a and Ctrl+x for increment/decrement
    elif key == 1:  # Ctrl+a
        cell = runner.get_current_cell()
        if cell.cell_type == CellType.INT:
            mult = max(1, runner.command_multiplier)
            current_value = cell.value if cell.value is not None else 0
            new_value = current_value + mult
            # Clear editing_value before setting content
            cell.editing_value = None
            cell.set_content(str(new_value))
            # Force display update with formats
            cell.editing_value = cell.get_display_value(
                column_format=runner.columns[runner.current_col].format,
                row_format=runner.rows[runner.current_row].format
            )
            runner.command_multiplier = 0
        elif cell.cell_type == CellType.BOOL:
            mult = max(1, runner.command_multiplier)
            current_value = cell.value if cell.value is not None else False
            # Flip the value modulo 2 for multiple commands
            new_value = not current_value if mult % 2 == 1 else current_value
            # Clear editing_value before setting content
            cell.editing_value = None
            cell.set_content(str(new_value).lower())
            # Force display update with formats
            cell.editing_value = cell.get_display_value(
                column_format=runner.columns[runner.current_col].format,
                row_format=runner.rows[runner.current_row].format
            )
            runner.command_multiplier = 0
        else:
            runner.error_message = "Error: Can only increment integer values or flip boolean values"
    elif key == 24:  # Ctrl+x
        cell = runner.get_current_cell()
        if cell.cell_type == CellType.INT:
            mult = max(1, runner.command_multiplier)
            current_value = cell.value if cell.value is not None else 0
            new_value = current_value - mult
            # Clear editing_value before setting content
            cell.editing_value = None
            cell.set_content(str(new_value))
            # Force display update with formats
            cell.editing_value = cell.get_display_value(
                column_format=runner.columns[runner.current_col].format,
                row_format=runner.rows[runner.current_row].format
            )
            runner.command_multiplier = 0
        elif cell.cell_type == CellType.BOOL:
            mult = max(1, runner.command_multiplier)
            current_value = cell.value if cell.value is not None else False
            # Flip the value modulo 2 for multiple commands
            new_value = not current_value if mult % 2 == 1 else current_value
            # Clear editing_value before setting content
            cell.editing_value = None
            cell.set_content(str(new_value).lower())
            # Force display update with formats
            cell.editing_value = cell.get_display_value(
                column_format=runner.columns[runner.current_col].format,
                row_format=runner.rows[runner.current_row].format
            )
            runner.command_multiplier = 0
        else:
            runner.error_message = "Error: Can only decrement integer values or flip boolean values"
    return True


def handle_command_mode(runner: CellRunner, key: int) -> bool:
    if key == Keys.ESC.value:
        runner.mode = OperationMode.NORMAL
        runner.current_input = ""
        runner.input_cursor_pos = 0
        runner.error_message = ""  # Clear error message when exiting command mode
    elif key == curses.KEY_BACKSPACE or key == 127:
        if runner.input_cursor_pos > 0:
            runner.current_input = (
                runner.current_input[: runner.input_cursor_pos - 1]
                + runner.current_input[runner.input_cursor_pos :]
            )
            runner.input_cursor_pos -= 1
            runner.error_message = ""  # Clear error message when editing command
    elif key == curses.KEY_LEFT:
        runner.input_cursor_pos = max(0, runner.input_cursor_pos - 1)
        runner.error_message = ""  # Clear error message on cursor movement
    elif key == curses.KEY_RIGHT:
        runner.input_cursor_pos = min(
            len(runner.current_input), runner.input_cursor_pos + 1
        )
        runner.error_message = ""  # Clear error message on cursor movement
    elif key == Keys.ENTER.value:
        current_command_input = runner.current_input
        runner.error_message = ""  # Clear any previous error message

        msg = f"command to run {current_command_input}"
        CURRENT_LOGGER.info(msg)

        split_command_input = current_command_input.split()
        if not split_command_input:  # Empty command
            runner.error_message = "Error: Empty command"
            return True

        first_command = split_command_input[0]

        try:
            if first_command == "w":
                destination = split_command_input[1]
                if destination:
                    normal_write(runner, destination)
                else:
                    normal_write(runner, "tmp.txt")

            elif first_command == "e":
                source = split_command_input[1]
                if source:
                    normal_read(runner, source)
                else:
                    normal_read(runner, "tmp.txt")

            elif first_command == "r":  # Row commands
                if len(split_command_input) < 2:
                    runner.error_message = "Error: Row command requires a subcommand"
                    return True
                subcommand = split_command_input[1]
                if subcommand == "h":  # Set row height
                    if len(split_command_input) < 3:
                        runner.error_message = "Error: Row height command requires a value"
                        return True
                    try:
                        height = int(split_command_input[2])
                        if height < 1:
                            runner.error_message = "Error: Row height must be at least 1"
                            return True
                        runner.rows[runner.current_row].set_height(height)
                    except ValueError:
                        runner.error_message = "Error: Row height must be a number"
                elif subcommand == "f":  # Set row format
                    if len(split_command_input) < 3:
                        runner.error_message = "Error: Row format command requires format type"
                        return True
                    format_type = split_command_input[2]
                    if format_type == "s":  # Separator format
                        separator = ""
                        if len(split_command_input) > 3:
                            # Handle quoted separator
                            separator = " ".join(split_command_input[3:])
                            if separator.startswith('"') and separator.endswith('"'):
                                separator = separator[1:-1]
                            elif separator.startswith("'") and separator.endswith("'"):
                                separator = separator[1:-1]
                        runner.rows[runner.current_row].set_format("s", separator)
                        # Force display update for all cells in the row
                        for col in range(len(runner.columns)):
                            cell = runner.rows[runner.current_row].get_cell(col)
                            if cell.value is not None:
                                # Clear editing_value before updating display
                                cell.editing_value = None
                                cell.editing_value = cell.get_display_value(
                                    column_format=runner.columns[col].format,
                                    row_format=runner.rows[runner.current_row].format
                                )
                    else:
                        runner.error_message = f"Error: Unknown format type '{format_type}'"
                else:
                    runner.error_message = f"Error: Unknown row subcommand '{subcommand}'"

            elif first_command == "c":  # Column commands
                if len(split_command_input) < 2:
                    runner.error_message = "Error: Column command requires a subcommand"
                    return True
                subcommand = split_command_input[1]
                if subcommand == "w":  # Set column width
                    if len(split_command_input) < 3:
                        runner.error_message = "Error: Column width command requires a value"
                        return True
                    try:
                        width = int(split_command_input[2])
                        if width < 1:
                            runner.error_message = "Error: Column width must be at least 1"
                            return True
                        runner.columns[runner.current_col].set_width(width)
                    except ValueError:
                        runner.error_message = "Error: Column width must be a number"
                elif subcommand == "f":  # Set column format
                    if len(split_command_input) < 3:
                        runner.error_message = "Error: Column format command requires format type"
                        return True
                    format_type = split_command_input[2]
                    if format_type == "s":  # Separator format
                        separator = ""
                        if len(split_command_input) > 3:
                            # Handle quoted separator
                            separator = " ".join(split_command_input[3:])
                            if separator.startswith('"') and separator.endswith('"'):
                                separator = separator[1:-1]
                            elif separator.startswith("'") and separator.endswith("'"):
                                separator = separator[1:-1]
                        runner.columns[runner.current_col].set_format("s", separator)
                        # Force display update for all cells in the column
                        for row in range(len(runner.rows)):
                            cell = runner.rows[row].get_cell(runner.current_col)
                            if cell.value is not None:
                                # Clear editing_value before updating display
                                cell.editing_value = None
                                cell.editing_value = cell.get_display_value(
                                    column_format=runner.columns[runner.current_col].format,
                                    row_format=runner.rows[row].format
                                )
                    else:
                        runner.error_message = f"Error: Unknown format type '{format_type}'"
                else:
                    runner.error_message = f"Error: Unknown column subcommand '{subcommand}'"

            elif first_command == "clear":
                normal_free(runner)

            elif first_command == "q":
                return False

            elif first_command == "it":  # Index tree command
                runner.index_table()
            elif first_command == "ic":  # Index current column
                runner.index_current_column()
            elif first_command == "ir":  # Index current row
                runner.index_current_row()
            elif first_command == "itx":  # Clear table index
                runner.clear_table_index()
            elif first_command == "icx":  # Clear column index
                runner.clear_column_index()
            elif first_command == "irx":  # Clear row index
                runner.clear_row_index()

            elif first_command == "*":  # Cell format command
                if len(split_command_input) < 3:
                    runner.error_message = "Error: Cell format command requires format type"
                    return True
                subcommand = split_command_input[1]
                if subcommand == "f":  # Set cell format
                    if len(split_command_input) < 3:
                        runner.error_message = "Error: Cell format command requires format type"
                        return True
                    format_type = split_command_input[2]
                    if format_type == "s":  # Separator format
                        separator = ""
                        if len(split_command_input) > 3:
                            # Handle quoted separator
                            separator = " ".join(split_command_input[3:])
                            if separator.startswith('"') and separator.endswith('"'):
                                separator = separator[1:-1]
                            elif separator.startswith("'") and separator.endswith("'"):
                                separator = separator[1:-1]
                        current_cell = runner.get_current_cell()
                        current_cell.set_format("s", separator)
                        # Force display update for the current cell
                        if current_cell.value is not None:
                            # Clear editing_value before updating display
                            current_cell.editing_value = None
                            current_cell.editing_value = current_cell.get_display_value(
                                column_format=runner.columns[runner.current_col].format,
                                row_format=runner.rows[runner.current_row].format
                            )
                    else:
                        runner.error_message = f"Error: Unknown format type '{format_type}'"
                else:
                    runner.error_message = f"Error: Unknown cell subcommand '{subcommand}'"

            elif first_command.isdigit():
                new_row = int(first_command)
                runner.move_to_row(new_row)
            else:
                runner.error_message = f"Error: Unknown command '{first_command}'"

        except Exception as e:
            runner.error_message = f"Error: {str(e)}"

        runner.mode = OperationMode.NORMAL
        runner.current_input = ""
        runner.input_cursor_pos = 0
    elif 32 <= key <= 126:  # Printable characters
        runner.current_input = (
            runner.current_input[: runner.input_cursor_pos]
            + chr(key)
            + runner.current_input[runner.input_cursor_pos :]
        )
        runner.input_cursor_pos += 1
        runner.error_message = ""  # Clear error message when typing new command

    return True


def handle_insert_mode(runner: CellRunner, key: int) -> bool:
    # Clear error message on any action in insert mode
    runner.error_message = ""
    
    if key == Keys.ESC.value:
        runner.mode = OperationMode.NORMAL
        runner.current_input = ""
        runner.input_cursor_pos = 0
        runner.get_current_cell().finish_editing()
        runner.suggestions = []
        runner.selected_suggestion = -1
        runner.suggestion_scroll_offset = 0
    elif key == curses.KEY_BACKSPACE or key == 127:
        if runner.input_cursor_pos > 0:
            runner.current_input = (
                runner.current_input[: runner.input_cursor_pos - 1]
                + runner.current_input[runner.input_cursor_pos :]
            )
            runner.input_cursor_pos -= 1
            # Create a temporary cell to check the new value
            temp_cell = Cell(runner.current_input)
            if temp_cell.value is not None:
                # Apply current formats to the new value
                runner.get_current_cell().editing_value = temp_cell.get_display_value(
                    column_format=runner.columns[runner.current_col].format,
                    row_format=runner.rows[runner.current_row].format
                )
            else:
                runner.get_current_cell().editing_value = runner.current_input
            # Update suggestions
            runner.suggestions = runner.get_suggestions(runner.current_input)
            runner.selected_suggestion = -1
            runner.suggestion_scroll_offset = 0
    elif key == curses.KEY_LEFT:
        runner.input_cursor_pos = max(0, runner.input_cursor_pos - 1)
    elif key == curses.KEY_RIGHT:
        runner.input_cursor_pos = min(
            len(runner.current_input), runner.input_cursor_pos + 1
        )
    elif key == Keys.ENTER.value:  # Enter
        if runner.selected_suggestion >= 0:
            runner.apply_suggestion()
        runner.get_current_cell().finish_editing()
        runner.mode = OperationMode.NORMAL
        runner.current_input = ""
        runner.input_cursor_pos = 0
        runner.suggestions = []
        runner.selected_suggestion = -1
        runner.suggestion_scroll_offset = 0
    elif key == ord('\t') or key == curses.KEY_DOWN:  # Tab or Down arrow
        if runner.suggestions:
            # If we're at the end of visible suggestions, scroll down
            if runner.selected_suggestion >= 0 and \
               runner.selected_suggestion >= runner.suggestion_scroll_offset + runner.visible_suggestions - 1:
                runner.scroll_suggestions(1)
            # Move to next suggestion
            runner.selected_suggestion = (runner.selected_suggestion + 1) % len(runner.suggestions)
            # Ensure selected suggestion is visible
            if runner.selected_suggestion < runner.suggestion_scroll_offset:
                runner.suggestion_scroll_offset = runner.selected_suggestion
            elif runner.selected_suggestion >= runner.suggestion_scroll_offset + runner.visible_suggestions:
                runner.suggestion_scroll_offset = runner.selected_suggestion - runner.visible_suggestions + 1
    elif key == curses.KEY_UP:  # Up arrow
        if runner.suggestions:
            # If we're at the start of visible suggestions, scroll up
            if runner.selected_suggestion >= 0 and \
               runner.selected_suggestion < runner.suggestion_scroll_offset:
                runner.scroll_suggestions(-1)
            # Move to previous suggestion
            runner.selected_suggestion = (runner.selected_suggestion - 1) % len(runner.suggestions)
            # Ensure selected suggestion is visible
            if runner.selected_suggestion < runner.suggestion_scroll_offset:
                runner.suggestion_scroll_offset = runner.selected_suggestion
            elif runner.selected_suggestion >= runner.suggestion_scroll_offset + runner.visible_suggestions:
                runner.suggestion_scroll_offset = runner.selected_suggestion - runner.visible_suggestions + 1
    elif 32 <= key <= 126:  # Printable characters
        # Check if this will change the cell type
        current_cell = runner.get_current_cell()
        old_type = current_cell.cell_type
        new_input = (
            runner.current_input[: runner.input_cursor_pos]
            + chr(key)
            + runner.current_input[runner.input_cursor_pos :]
        )
        
        # Create a temporary cell to check the new type
        temp_cell = Cell(new_input)
        
        # Only warn if changing from a non-empty type to a different type
        if (old_type != CellType.EMPTY and 
            temp_cell.cell_type != old_type and 
            temp_cell.cell_type != CellType.EMPTY):
            runner.error_message = f"Warning: Changing cell type from {old_type} to {temp_cell.cell_type}"
        
        runner.current_input = new_input
        runner.input_cursor_pos += 1
        
        # Apply current formats to the new value
        if temp_cell.value is not None:
            current_cell.editing_value = temp_cell.get_display_value(
                column_format=runner.columns[runner.current_col].format,
                row_format=runner.rows[runner.current_row].format
            )
        else:
            current_cell.editing_value = new_input
            
        # Update suggestions
        runner.suggestions = runner.get_suggestions(runner.current_input)
        runner.selected_suggestion = -1
        runner.suggestion_scroll_offset = 0

    return True


def handle_visual_mode(runner: CellRunner, key: int) -> bool:
    if key == Keys.ESC.value:
        runner.mode = OperationMode.NORMAL
    elif key == ord("j"):
        mult = max(1, runner.command_multiplier)
        runner.current_row = min(len(runner.rows) - 1, runner.current_row + mult)
        runner.visual_end_row = runner.current_row
        runner.command_multiplier = 0
    elif key == ord("k"):
        mult = max(1, runner.command_multiplier)
        runner.current_row = max(0, runner.current_row - mult)
        runner.visual_end_row = runner.current_row
        runner.command_multiplier = 0
    elif key == ord("h"):
        mult = max(1, runner.command_multiplier)
        runner.current_col = max(0, runner.current_col - mult)
        runner.visual_end_col = runner.current_col
        runner.command_multiplier = 0
    elif key == ord("l"):
        mult = max(1, runner.command_multiplier)
        runner.current_col = min(
            len(runner.rows[0].cells) - 1,
            runner.current_col + mult,
        )
        runner.visual_end_col = runner.current_col
        runner.command_multiplier = 0
    elif key == ord("x"):
        # Clear all cells in the visual selection
        min_row = min(runner.visual_start_row, runner.visual_end_row)
        max_row = max(runner.visual_start_row, runner.visual_end_row)
        min_col = min(runner.visual_start_col, runner.visual_end_col)
        max_col = max(runner.visual_start_col, runner.visual_end_col)
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                runner.rows[row].get_cell(col).set_content("")
        
        # Exit visual mode after deletion
        runner.mode = OperationMode.NORMAL
    elif ord("0") <= key <= ord("9"):
        runner.command_multiplier = runner.command_multiplier * 10 + (key - ord("0"))
    return True


def main(stdscr: CursesWindow) -> None:
    # Initialize color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal mode
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Insert mode
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Rulers
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)

    # Hide the cursor
    curses.curs_set(0)

    # Initialize the cell runner
    runner = CellRunner()
    while True:
        display_grid(stdscr, runner)
        key = stdscr.getch()
        if not handle_input(runner, key):
            break


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        sys.exit(0)
