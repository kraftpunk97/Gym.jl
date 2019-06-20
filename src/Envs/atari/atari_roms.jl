(_, _, files), _ = iterate(walkdir(@__DIR__ * "/atari_roms"))
game_files = [endswith(filename, ".bin") for filename in files]
