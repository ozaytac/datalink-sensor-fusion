"""
Simulation history — records per-step state for replay/export.
"""

from __future__ import annotations
from typing import Any, Optional


class SimulationHistory:
    """
    Collects simulation snapshots for later replay or export.

    Each call to :meth:`append` stores one time-step of data.
    A sequential iterator (:meth:`next`) supports replay loops.
    """

    def __init__(self) -> None:
        self.pp_poses:       list = []
        self.dets:           list = []
        self.tracks:         list = []
        self.mpar_pos:       list = []
        self.mpar_ori:       list = []
        self.offset:         list = []
        self.covarage_config: list = []
        self.n_element: int = 0
        self._index:    int = 0

    # ------------------------------------------------------------------ #
    def append(
        self,
        pp_poses: Any,
        dets: Any,
        tracks: Any,
        mpar_pos: Any,
        mpar_ori: Any,
        offset: Any,
        covarage_config: Any,
    ) -> None:
        self.pp_poses.append(pp_poses)
        self.dets.append(dets)
        self.tracks.append(tracks)
        self.mpar_pos.append(mpar_pos)
        self.mpar_ori.append(mpar_ori)
        self.offset.append(offset)
        self.covarage_config.append(covarage_config)
        self.n_element += 1

    # ------------------------------------------------------------------ #
    def reset_index(self) -> None:
        self._index = 0

    # ------------------------------------------------------------------ #
    def next(self) -> tuple:
        """
        Return the next snapshot and advance the cursor.

        Returns
        -------
        (pp_poses, dets, tracks, mpar_pos, mpar_ori, offset,
         covarage_config, is_end)
        """
        if self._index >= self.n_element:
            return None, None, None, None, None, None, None, True

        i = self._index
        self._index += 1
        return (
            self.pp_poses[i],
            self.dets[i],
            self.tracks[i],
            self.mpar_pos[i],
            self.mpar_ori[i],
            self.offset[i],
            self.covarage_config[i],
            False,
        )

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.n_element

    def __repr__(self) -> str:
        return f"SimulationHistory(n_element={self.n_element}, index={self._index})"
