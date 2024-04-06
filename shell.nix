{}:

let
  pin = rec {
    commit = "1487bdea619e4a7a53a4590c475deabb5a9d1bfb"; # nixos-23.11 @ 2023-04-06
    pkgsSrc = builtins.fetchTarball {
      url = "https://github.com/NixOS/nixpkgs/archive/${commit}.tar.gz";
      # Use "nix-prefetch-url --unpack <url>" to calculate sha256
      # Or set to empty and wait for the error to tell you the right one
      sha256 = "0aga80czsq950mja7hrdamm02lfhgpnd47ak2a3a2zqvxasbizaw";
    };
    pkgs = import pkgsSrc {};
  };

in

pin.pkgs.mkShell {
  name = "rps";
  buildInputs = with pin.pkgs; [
    (python311.withPackages (ps: with ps; [
      # By default, opencv4 is built without GUI
      # https://stackoverflow.com/a/68700756/360390
      (ps.opencv4.override {
        enableGtk3 = true;
      })
    ]))

    poetry
    # Poetry has a bug where it doesn't pick current active python version
    # https://github.com/python-poetry/poetry/issues/9278
    # Hence we need to install *any* python package to populate PYTHONPATH
    python311Packages.structlog

    # Bringing ML-related stuff through nix because PyPi ones segfault, mismatch versions, etc.
    # WARNING: Stick with 23.11 NixOS branch! Newer versions comes with new version fo tensorflow/keras
    #          which have break API changes!
    python311Packages.tensorflow
    python311Packages.keras
    python311Packages.dm-tree
    python311Packages.rich
    python311Packages.ml-dtypes


    # These + the LD_LIBRARY_PATH are requires for mediapipe which we insall through poetry
    zlib
    libGL
    glib
  ];

   LD_LIBRARY_PATH = "${pin.pkgs.zlib}/lib:${pin.pkgs.stdenv.cc.cc.lib}/lib:${pin.pkgs.libGL}/lib:${pin.pkgs.glib.out}/lib:/run/opengl-driver/lib";
}
