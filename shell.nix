with ( import (builtins.fetchTarball https://channels.nixos.org/nixpkgs-23.05-darwin/nixexprs.tar.xz) {});
mkShell {
  buildInputs = [
    poetry
    python311
  ];
  shellHook = ''
    # Set up virtual environment
    if [ ! -d .venv ]; then
        echo "No .venv found, creating..."
        python -m venv .venv
    fi;
    source .venv/bin/activate

    export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
