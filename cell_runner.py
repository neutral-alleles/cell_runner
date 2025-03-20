#!/usr/bin/env python3
import csv
import curses
import logging
import sys
from enum import Enum
from typing import Any, List, Set, Tuple, TypeAlias

CursesWindow: TypeAlias = Any  # Represents a curses window object

CELL_WIDTH: int = 16
CELL_REF_SEPARATOR = " :: "


class OperationMode(Enum):
    NORMAL = 1
    INSERT = 2
    COMMAND = 4

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


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


class Cell:
    def __init__(self, content: str = ""):
        self.content = content
        self.cursor_pos = 0
        self.formula = None
        self.value = None
        self.dependencies: Set[Tuple[int, int]] = set()  # Set of (row, col) tuples
        self.dependents: Set[Tuple[int, int]] = set()  # Cells that depend on this cell

    def set_content(self, content: str):
        self.content = content
        if content.startswith("="):
            self.formula = content[1:]
            self.dependencies = set()
            self.value = None
        else:
            self.formula = None
            try:
                self.value = float(content) if content else 0
            except ValueError:
                self.value = content

    def get_display_value(self) -> str:
        if self.formula is not None:
            return f"={self.formula}"
        if isinstance(self.value, float):
            return f"{self.value:.2f}"
        return str(self.value) if self.value is not None else ""


class CellRunner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid: List[List[Cell]] = [[Cell()]]  # 2D grid of cells
        self.current_row: int = 0
        self.current_col: int = 0
        self.mode = OperationMode.NORMAL
        self.command_buffer = ""
        self.command_multiplier: int = 0
        self.current_input = ""
        self.input_cursor_pos = 0
        self.columns = 26  # A-Z

    def get_current_cell(self) -> Cell:
        return self.grid[self.current_row][self.current_col]

    def get_cell_reference(self, row: int, col: int) -> str:
        return f"{self.get_column_letter(col)}{row + 1}"

    def get_column_letter(self, col: int) -> str:
        if col < 26:
            return chr(ord("A") + col)
        else:
            return chr(ord("A") + (col // 26) - 1) + chr(ord("A") + (col % 26))

    def ensure_grid_size(self, row: int, col: int):
        # Ensure enough rows
        while len(self.grid) <= row:
            self.grid.append([])

        # Ensure enough columns in each row
        for r in range(len(self.grid)):
            while len(self.grid[r]) <= col:
                self.grid[r].append(Cell())

    def add_cell_right(self):
        self.ensure_grid_size(self.current_row, self.current_col + 1)
        self.current_col += 1

    def add_cell_below(self):
        self.ensure_grid_size(self.current_row + 1, self.current_col)
        self.current_row += 1

    def add_cell_above(self):
        self.grid.insert(self.current_row, [Cell() for _ in range(len(self.grid[0]))])

    def clear_cell(self):
        self.get_current_cell().set_content("")

    def delete_cell(self):
        if len(self.grid) > 1 or len(self.grid[0]) > 1:
            # Remove the current cell
            self.grid[self.current_row].pop(self.current_col)

            # If the row is empty, remove it
            if not self.grid[self.current_row]:
                self.grid.pop(self.current_row)

            # Adjust current position
            self.current_col = max(0, self.current_col - 1)
            self.current_row = min(self.current_row, len(self.grid) - 1)

    def find_last_non_empty_in_row(self, row: int) -> int:
        for col in range(len(self.grid[row]) - 1, -1, -1):
            if self.grid[row][col].content:
                return col
        return -1

    def update_cell_values(self):
        return

    def move_to_row(self, new_row: int):
        self.ensure_grid_size(new_row, self.current_col)
        self.current_row = new_row - 1

    def move_to_cell(self, new_cell: int):
        self.ensure_grid_size(self.current_row, new_cell)
        self.current_cell = new_cell


def normal_write(runner: CellRunner, destination: str):
    # TODO add ranges and different formats

    with open(destination, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        content_grid = [[cell.content for cell in row] for row in runner.grid]
        writer.writerows(content_grid)


def normal_free(runner: CellRunner):
    runner.reset()


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

        runner.grid = new_grid


def main(stdscr: CursesWindow) -> None:
    # Initialize color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Normal mode
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Insert mode
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Rulers
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)

    # Hide the cursor
    curses.curs_set(0)

    # Initialize the cell runner
    runner = CellRunner()

    # Main loop
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        height = int(height)
        width = int(width)

        horizontal_capacity: int = width // (CELL_WIDTH)

        horizontal_start: int = runner.current_col - horizontal_capacity // 2
        horizontal_end: int = runner.current_col + horizontal_capacity // 2

        if horizontal_start < 0:
            horizontal_start: int = 0
            horizontal_end: int = horizontal_capacity

        vertical_capacity = height // 2

        vertical_start: int = runner.current_row - vertical_capacity // 2
        vertical_end: int = runner.current_row + vertical_capacity // 2

        if vertical_start < 0:
            vertical_start: int = 0
            vertical_end: int = vertical_capacity

        # Update cell values before display
        runner.update_cell_values()

        # Draw horizontal ruler (column letters)
        ruler_y = 0
        for col in range(runner.columns):
            if horizontal_start > col:
                continue
            elif col > horizontal_end:
                break

            col_letter = runner.get_column_letter(col)
            x = (col - horizontal_start) * CELL_WIDTH + 2
            if x < width - 1:
                if runner.current_col == col:
                    stdscr.addstr(ruler_y, x, col_letter, curses.color_pair(4))
                else:
                    stdscr.addstr(ruler_y, x, col_letter, curses.color_pair(1))

        row_num_len = len(str(vertical_end))
        border = row_num_len + 1

        # Draw vertical ruler (row numbers) and cells
        for i, row in enumerate(runner.grid):
            row_num = str(i + 1)
            if i < vertical_start:
                continue
            elif i > vertical_end:
                break

            y = (
                i - vertical_start
            ) * 2 + 1  # Each cell takes 3 lines, starting after the ruler
            if y >= height - 1:
                break

            # Draw row number
            if runner.current_row == i:
                stdscr.addstr(y, 0, row_num.rjust(border), curses.color_pair(4))
            else:
                stdscr.addstr(y, 0, row_num.rjust(border), curses.color_pair(1))

            # Draw cell border
            stdscr.addstr(y, border, "│")
            stdscr.addstr(y, width - 1, "│")
            stdscr.addstr(y + 1, 0, "─" * (width - 1))

            # Draw cells in the row
            for j, cell in enumerate(row):
                if j < horizontal_start:
                    continue
                elif j > horizontal_end:
                    break

                x = (j - horizontal_start) * CELL_WIDTH + 1 + border

                if x >= width - 2:
                    break

                content = cell.get_display_value()
                if i == runner.current_row and j == runner.current_col:
                    if not content:
                        content = CELL_WIDTH * "_"
                    stdscr.addstr(y, x, content[:7], curses.color_pair(4))
                else:
                    stdscr.addstr(y, x, content[:7])

        # Display mode and command buffer
        cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
        mode_str = f"Mode: {runner.mode}"
        if runner.command_buffer or runner.command_multiplier:
            mode_str += (
                f" | Command: {runner.command_multiplier}{runner.command_buffer}"
            )

        # Display input line in insert mode
        if runner.mode == OperationMode.INSERT:
            cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
            input_str = f"{runner.current_input}"
            if runner.input_cursor_pos <= len(input_str):
                input_str = (
                    cell_ref
                    + CELL_REF_SEPARATOR
                    + input_str[: runner.input_cursor_pos]
                    + "█"
                    + input_str[runner.input_cursor_pos :]
                )
            final_input_str = f"{mode_str} {input_str}"
            stdscr.addstr(height - 1, 0, final_input_str, curses.color_pair(2))
        elif runner.mode == OperationMode.COMMAND:
            cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
            input_str = f"{runner.current_input}"
            if runner.input_cursor_pos <= len(input_str):
                input_str = (
                    ":"
                    + input_str[: runner.input_cursor_pos]
                    + "█"
                    + input_str[runner.input_cursor_pos :]
                )
            final_input_str = f"{mode_str} {input_str}"
            stdscr.addstr(height - 1, 0, final_input_str, curses.color_pair(2))
        else:
            cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
            cell_content = runner.get_current_cell().content
            normal_str = f"{mode_str} {cell_ref} {cell_content}"
            stdscr.addstr(height - 1, 0, normal_str, curses.color_pair(1))

        # Refresh the screen
        stdscr.refresh()

        # Get user input
        key: int = int(stdscr.getch())

        if runner.mode == OperationMode.NORMAL:
            if ord("0") <= key <= ord("9"):
                runner.command_multiplier = runner.command_multiplier * 10 + (
                    key - ord("0")
                )
            elif key == ord("j"):
                mult = max(1, runner.command_multiplier)
                runner.current_row = min(
                    len(runner.grid) - 1, runner.current_row + mult
                )
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
                    len(runner.grid[0]) - 1,
                    runner.current_col + mult,
                )
                runner.command_multiplier = 0
            elif key == ord("i"):
                runner.mode = OperationMode.INSERT
                runner.current_input = runner.get_current_cell().content
                runner.input_cursor_pos = len(runner.current_input)
            elif key == ord("a"):
                runner.add_cell_right()
                runner.mode = OperationMode.INSERT
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("A"):
                last_col = runner.find_last_non_empty_in_row(runner.current_row)
                if last_col >= 0:
                    runner.current_col = last_col + 1
                    runner.ensure_grid_size(runner.current_row, runner.current_col)
                runner.mode = OperationMode.INSERT
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("o"):
                runner.add_cell_below()
                runner.mode = OperationMode.INSERT
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("O"):
                runner.add_cell_above()
                runner.mode = OperationMode.INSERT
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("x"):
                runner.clear_cell()
            elif key == ord("d"):
                if runner.command_buffer == "d":
                    runner.delete_cell()
                    runner.command_buffer = ""
                else:
                    runner.command_buffer = "d"
            elif key == ord(":"):
                runner.mode = OperationMode.COMMAND
            elif key == 27:  # ESC
                runner.command_buffer = ""
                runner.command_multiplier = 0

        elif runner.mode == OperationMode.COMMAND:
            if key == 27:
                runner.mode = OperationMode.NORMAL
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == curses.KEY_BACKSPACE or key == 127:
                if runner.input_cursor_pos > 0:
                    runner.current_input = (
                        runner.current_input[: runner.input_cursor_pos - 1]
                        + runner.current_input[runner.input_cursor_pos :]
                    )
                    runner.input_cursor_pos -= 1
            elif key == curses.KEY_LEFT:
                runner.input_cursor_pos = max(0, runner.input_cursor_pos - 1)
            elif key == curses.KEY_RIGHT:
                runner.input_cursor_pos = min(
                    len(runner.current_input), runner.input_cursor_pos + 1
                )
            elif key == 10:  # Enter
                current_command_input = runner.current_input

                msg = f"command to run {current_command_input}"
                CURRENT_LOGGER.info(msg)

                split_command_input = current_command_input.split()

                first_command = split_command_input[0]

                if first_command == "w":
                    destination = split_command_input[1]
                    if destination:
                        normal_write(runner, destination)
                    else:
                        normal_write(runner, "tmp.txt")

                elif first_command == "r":
                    source = split_command_input[1]
                    if source:
                        normal_read(runner, source)
                    else:
                        normal_read(runner, "tmp.txt")

                elif first_command == "clear":
                    normal_free(runner)

                elif first_command == "q":
                    break

                elif first_command.isdigit():
                    new_row = int(first_command)
                    runner.move_to_row(new_row)

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

        elif runner.mode == OperationMode.INSERT:
            if key == 27:  # ESC
                runner.mode = OperationMode.NORMAL
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == curses.KEY_BACKSPACE or key == 127:
                if runner.input_cursor_pos > 0:
                    runner.current_input = (
                        runner.current_input[: runner.input_cursor_pos - 1]
                        + runner.current_input[runner.input_cursor_pos :]
                    )
                    runner.input_cursor_pos -= 1
            elif key == curses.KEY_LEFT:
                runner.input_cursor_pos = max(0, runner.input_cursor_pos - 1)
            elif key == curses.KEY_RIGHT:
                runner.input_cursor_pos = min(
                    len(runner.current_input), runner.input_cursor_pos + 1
                )
            elif key == 10:  # Enter
                runner.get_current_cell().set_content(runner.current_input)
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


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        sys.exit(0)

