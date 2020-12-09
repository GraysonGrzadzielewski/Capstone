def main():
    import retro
    from pathlib import Path as p
    import shutil

    retro_dir =  p(retro.__file__).parents[0] / 'data/stable/SuperMarioBros-Nes/'
    files_to_copy=[
        p('./integration/data.json').resolve(),
        p('./integration/metadata.json').resolve(),
        p('./integration/scenario.json').resolve()
    ]

    for f in files_to_copy:
        shutil.copy(str(f), str(retro_dir))
if __name__ == '__main__':
    main()
