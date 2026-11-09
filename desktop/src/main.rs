// Prevents a console window from flashing when the app is launched from a
// terminal; the Windows GUI subsystem is the right one for a desktop shell.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tau::run();
}
