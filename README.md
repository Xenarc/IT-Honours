# GNU Radio environment

## Installation

WSL2
`sudo apt-get install gnuradio`
Follow `https://wiki.gnuradio.org/index.php?title=WindowsInstall#WSL_|_Ubuntu`

## Running

1. Run
`"C:\Program Files\VcXsrv\xlaunch.exe"`
  Make sure that you start with `Disable Access Control` checked:
2. Open WSL Ubuntu 22.04 in Windows Terminal
3. Run `gnuradio-companion`

## Development

Create a module:
`gr_modtool newmod <name>`

Create a new block:
`gr_modtool add <name>`

Location:
`cd /mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio`

Compile:
`cd build`
`cmake ..`
`make`
`sudo make install`
`sudo ldconfig`

Or more easily, just run the build.sh in WSL2, or in vscode debug and run

## Debugging

If you come up with an error like `EODNENT Cannot open \\pipe\ASiD_@*#123`, simply restart vscode.
