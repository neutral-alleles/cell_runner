#!/usr/bin/env python3
import curses
import re
import sys
from typing import List, Optional, Set, Tuple

import logging

CELL_WIDTH = 16
CELL_REF_SEPARATOR = " :: "

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Function to set up a logger with file and console handlers"""
    
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler(log_file)  # File handler
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

CURRENT_LOGGER: logging.Logger = setup_logger('CELL RUNNER', 'log.txt')


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
        self.grid: List[List[Cell]] = [[Cell()]]  # 2D grid of cells
        self.current_row = 0
        self.current_col = 0
        self.mode = "normal"  # "normal" or "insert"
        self.command_buffer = ""
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


def main(stdscr) -> None:
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

        # Update cell values before display
        runner.update_cell_values()

        # Draw horizontal ruler (column letters)
        ruler_y = 0
        for col in range(runner.columns):
            col_letter = runner.get_column_letter(col)
            x = col * CELL_WIDTH + 2 
            if x < width - 1:
                if runner.current_col == col:
                    stdscr.addstr(ruler_y, x, col_letter, curses.color_pair(4))
                else:
                    stdscr.addstr(ruler_y, x, col_letter, curses.color_pair(1))

        # Draw vertical ruler (row numbers) and cells
        for i, row in enumerate(runner.grid):
            y = i * 3 + 1  # Each cell takes 3 lines, starting after the ruler
            if y >= height - 1:
                break

            # Draw row number
            row_num = str(i + 1)
            if runner.current_row == i:
                stdscr.addstr(y, 0, row_num.rjust(2), curses.color_pair(4))
            else:
                stdscr.addstr(y, 0, row_num.rjust(2), curses.color_pair(1))

            # Draw cell border
            stdscr.addstr(y, 2, "│")
            stdscr.addstr(y, width - 1, "│")
            stdscr.addstr(y + 1, 0, "─" * width)

            # Draw cells in the row
            for j, cell in enumerate(row):
                x = j * CELL_WIDTH + 3
                if x >= width - 1:
                    break

                content = cell.get_display_value()
                if i == runner.current_row and j == runner.current_col:
                    stdscr.addstr(y + 1, x, content[:7], curses.color_pair(4))
                else:
                    stdscr.addstr(y + 1, x, content[:7], curses.color_pair(1))

        # Display mode and command buffer
        mode_str = f"Mode: {runner.mode}"
        if runner.command_buffer:
            mode_str += f" | Command: {runner.command_buffer}"

        # Display input line in insert mode
        if runner.mode == "insert":
            cell_ref = runner.get_cell_reference(runner.current_row, runner.current_col)
            input_str = f"{runner.current_input}"
            if runner.input_cursor_pos <= len(input_str):
                input_str = (
                    cell_ref + CELL_REF_SEPARATOR + 
                    input_str[: runner.input_cursor_pos]
                    + "█"
                    + input_str[runner.input_cursor_pos :]
                )
            stdscr.addstr(height - 1, 0, input_str, curses.color_pair(2))
        else:
            stdscr.addstr(height - 1, 0, mode_str, curses.color_pair(1))

        # Refresh the screen
        stdscr.refresh()

        # Get user input
        key = stdscr.getch()

        if runner.mode == "normal":
            if key == ord("j"):
                runner.current_row = min(len(runner.grid) - 1, runner.current_row + 1)
            elif key == ord("k"):
                runner.current_row = max(0, runner.current_row - 1)
            elif key == ord("h"):
                runner.current_col = max(0, runner.current_col - 1)
            elif key == ord("l"):
                runner.current_col = min(
                    len(runner.grid[0]) - 1, runner.current_col + 1
                )
            elif key == ord("i"):
                runner.mode = "insert"
                runner.current_input = runner.get_current_cell().content
                runner.input_cursor_pos = len(runner.current_input)
            elif key == ord("a"):
                runner.add_cell_right()
                runner.mode = "insert"
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("A"):
                last_col = runner.find_last_non_empty_in_row(runner.current_row)
                if last_col >= 0:
                    runner.current_col = last_col + 1
                    runner.ensure_grid_size(runner.current_row, runner.current_col)
                runner.mode = "insert"
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("o"):
                runner.add_cell_below()
                runner.mode = "insert"
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("O"):
                runner.add_cell_above()
                runner.mode = "insert"
                runner.current_input = ""
                runner.input_cursor_pos = 0
            elif key == ord("d"):
                if runner.command_buffer == "d":
                    runner.delete_cell()
                    runner.command_buffer = ""
                else:
                    runner.command_buffer = "d"
            elif key == ord("q"):
                break
            elif key == 27:  # ESC
                runner.command_buffer = ""

        elif runner.mode == "insert":
            if key == 27:  # ESC
                runner.mode = "normal"
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
                runner.mode = "normal"
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

