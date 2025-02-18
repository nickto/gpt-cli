with ( import (builtins.fetchTarball https://channels.nixos.org/nixos-24.11/nixexprs.tar.xz) {});
mkShell {
  buildInputs = [
    uv
  ];
  shellHook = ''
    # Set up virtual environment
    if [ ! -d .venv ]; then
        echo "No .venv found, creating..."
        uv venv .venv --python 3.11
    fi;
    source .venv/bin/activate
  '';
}
