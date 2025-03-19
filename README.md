# Cell Runner

A vim-like terminal-based cell editor that allows you to navigate between cells and edit their contents, with Excel-like formula support.

## Requirements

- Python 3.6 or higher
- curses library (usually comes with Python)

## Installation

No installation required. Just run the script directly:

```bash
python cell_runner.py
```

## Usage

The program operates in two modes: normal mode and insert mode.

### Normal Mode Commands

- `h` - Move to the left cell
- `j` - Move to the cell below
- `k` - Move to the cell above
- `l` - Move to the right cell
- `i` - Enter insert mode at cursor position
- `a` - Insert a new cell to the right and enter insert mode
- `A` - Move to the right of the last non-empty cell in the current row and enter insert mode
- `o` - Create a new cell below and enter insert mode
- `O` - Create a new cell above and enter insert mode
- `dd` - Delete current cell
- `q` - Quit the program
- `ESC` - Clear command buffer

### Insert Mode Commands

- `ESC` - Cancel editing and return to normal mode
- `Enter` - Confirm changes and return to normal mode
- Arrow keys - Move cursor in the input line
- Backspace - Delete character before cursor
- Any printable character - Insert at cursor position

### Formula Support

Cells support Excel-like formulas:

1. Start a formula with `=` to make it a formula cell
2. Use cell references in the format `A1`, `B2`, etc., where:
   - Letters (A, B, C..., Z, AA, AB...) represent columns
   - Numbers (1, 2, 3...) represent rows
3. Basic arithmetic operations are supported:
   - Addition: `+`
   - Subtraction: `-`
   - Multiplication: `*`
   - Division: `/`
   - Parentheses for grouping: `()`

Examples:
- `=A1 + B2` - Add values from cells A1 and B2
- `=(A1 * B2) / C3` - Multiply A1 and B2, then divide by C3
- `=A1 + 5` - Add 5 to the value in A1
- `=AA1 + Z2` - Add values from cells AA1 and Z2

## Visual Indicators

- The current cell is highlighted with square brackets `[]`
- In insert mode, the input line shows at the bottom of the screen in the format `A1 :: <value>`
- The cursor in insert mode is shown as a block character `█` in the input line
- The current mode is displayed at the bottom of the screen (in normal mode)
- Command buffer is shown when entering multi-character commands (like `dd`)
- Formula cells show both the formula and its calculated value
- Numeric values are displayed with 2 decimal places
- Column letters (A-Z, AA-ZZ) are shown in yellow at the top
- Row numbers are shown in yellow on the left side
- Cell borders are drawn with box-drawing characters 