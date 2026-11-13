# app/work — your own programs

This folder is yours. Git ignores everything in it except this README and
`.gitignore`, so `git pull` never touches your files and they never end up in
SAM's history.

Start from one of the programs in `../tutorial/` (see its README) and edit the
copy, never the original:

```sh
cp app/tutorial/MatrixOperatorJT.cpp app/work/Li6_spectrum.cpp   # any name
# edit the settings at the top of app/work/Li6_spectrum.cpp
make -j 8 apps
mpirun -np 1 bin/work/Li6_spectrum
```

Every `.cpp` file directly in `app/work/` is built into `bin/work/`. Files in
subfolders are not built, so you can keep notes, inputs or old versions there.

To keep your work in sync between machines, you can make this folder a git
repository of its own (`git init` here, or clone a private repository into
it). SAM's git ignores it either way.
