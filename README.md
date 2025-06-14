# gemma-2-2b-it.cs
gemma-2-2b-it int8 cpu inference in one file of pure C#

I used Gemini 2.5 Pro Preview to port this Rust repo: https://github.com/samuel-vitorino/lm.rs/tree/24c45b01419e1d7989dfc93ac6209b13e6edbbfa to C#. So all credits go to creator of original repo and to Gemini 2.5 Pro Preview. You should use python scripts from original repo to create llm and tokenizer files.Then move them to the folder where you placed app's release files.

You can build project by double clicking on build.bat

To run project you should double click on launch.bat. By default it supposed that you called llm file "model.bin". Otherwise you will need to edit launch.but accordingly.
