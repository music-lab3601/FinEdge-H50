"""Immutable progress checkpoints; publish result.json once at completion.

Windows readers may deny DELETE sharing, preventing replacement of an open
file. Never replace the report they may already be reading during a run.
No retries of hardware operations or data visibility delays are involved.
"""
from software.runtime.delivery import write_json


class RunReports:
    def __init__(self, directory):
        self.directory=directory
        self.checkpoints=directory/'checkpoints'
        self.checkpoints.mkdir(exist_ok=False)
        self.sequence=0
        self.finished=False

    def save(self, report, *, final=False):
        if self.finished:raise RuntimeError('Completed report must not be overwritten')
        self.sequence+=1
        checkpoint=self.checkpoints/f'report_{self.sequence:06d}.json'
        if checkpoint.exists():raise FileExistsError(checkpoint)
        write_json(checkpoint,report)
        if final:
            result=self.directory/'result.json'
            if result.exists():raise FileExistsError(result)
            write_json(result,report)
            self.finished=True
        return checkpoint
