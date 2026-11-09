"""Review/physical-session gates, evaluated before any board connection."""
from shared.paths import resolve_path
from dataclasses import dataclass
from pathlib import Path
from shared.model_paths import ROOT
from software.runtime.delivery import read_json, sha256


PLAN='artifacts/fpga_udw_followup/E_plan_gapfix_20261024_01/plan.json'
CRITICAL_FILES=(
    'src/finedge_host/udw_e_release.py','src/finedge_host/udw_e_board.py',
    'src/finedge_host/udw_e_inputs.py','src/finedge_host/udw_e_candidate.py',
    'src/finedge_host/udw_e_preflight.py','src/finedge_host/udw_e_frontend.py',
    'src/finedge_host/udw_e_offline.py','src/finedge_host/udw_measurement_plan.py',
    'src/finedge_host/udw_lab.py','src/finedge_host/udw_runtime.py',
    'src/finedge_host/ddr_transport_contract.py','src/finedge_host/stream.py',
    'src/finedge_host/backends.py','src/finedge_host/delivery.py','src/finedge_host/paths.py',
    'src/finedge_host/udw_reports.py','src/finedge_host/board.py',
    'scripts/udw_e_board_check.py','scripts/udw_owner_daemon.py',
    'hardware/wujing_fpai/software/udw_lab/owner.cpp',
    'hardware/wujing_fpai/software/udw_lab/owner_startup.h',
    'build_udw_lab/Release/finedge_udw_owner.exe',
    'artifacts/icraft/tcn_ed4_ZG.json','artifacts/icraft/tcn_ed4_ZG.raw',
    'artifacts/export/tcn_ed4.onnx','vendor/ed4_model/normalization.json',
    'vendor/ed4_model/package_manifest.json',
    'artifacts/fpga_user_ddr_window/boot_20261024_02/BOOT.bin',PLAN)
SDK_FILES=tuple(str(resolve_path('tool://icraft/bin/'))+name for name in (
    'icraft_xrt.dll','icraft_xir.dll','icraft_hostbackend.dll','icraft_zg330backend.dll',
    'socketzg330aiu.dll','irpc_core.dll','irpc_port_tcp.dll','torch_cpu_ic.dll','c10_ic.dll'))
CONSTRAINTS=dict(records=448,windows=22,batch=64,emit=16,warmup=10,quota_bytes=524288,
                 reuse=False,automatic_reset=False,automatic_owner_termination=False)
DEVICE_VERSIONS=dict(device='25122301',icore='FMSHZGV3TECH-AID - 24160628')


def workspace_file(path):
    result=Path(path)
    if not result.is_absolute():result=ROOT/result
    result=result.resolve()
    if not result.is_relative_to(ROOT) or not result.is_file():
        raise ValueError('Review/confirmation file must be in the active project')
    return result


@dataclass(frozen=True)
class BoardPermit:
    decision_path: Path
    decision_sha256: str
    reviewed_files: dict
    reviewed_sdk_files: dict

    def validate(self,arm):
        if arm not in ('S','H'):raise ValueError('Only a single S or H arm per cold session')
        if sha256(self.decision_path)!=self.decision_sha256:raise ValueError('Review decision changed')
        decision=read_json(self.decision_path)
        if decision.get('authority_kind')=='direct_user_deferred_review':
            if (decision.get('user_authorized_bounded_execution') is not True
                or decision.get('review_deferred') is not True
                or decision.get('independent_review_passed') is not False
                or decision.get('user_instruction')!='暂时无需审核，等待此轮结束后一起审核'
                or decision.get('preceding_user_task')!='进行下一阶段'):
                raise ValueError('Explicit bounded user authorization/deferred-review record is missing')
        elif (decision.get('E_board_experiment_allowed') is not True
              or decision.get('candidate_validation_accepted') is not True
              or 'UDW_E_PLAN_REVIEW_01' not in decision.get('closed_findings',[])):
            raise ValueError('Independent E candidate/board release is missing')
        if decision.get('constraints')!=CONSTRAINTS or arm not in decision.get('allowed_arms',[]):
            raise ValueError('Unreviewed workload/arm')
        if decision.get('expected_device_versions')!=DEVICE_VERSIONS:
            raise ValueError('Unreviewed device/core version')
        if decision.get('reviewed_files')!=self.reviewed_files:
            raise ValueError('Review file map changed')
        if not set(CRITICAL_FILES)<=set(self.reviewed_files):raise ValueError('Incomplete reviewed source/artifact hashes')
        for name,h in self.reviewed_files.items():
            p=workspace_file(name)
            if sha256(p)!=h:raise ValueError('Reviewed file changed: '+name)
        if (decision.get('reviewed_sdk_files')!=self.reviewed_sdk_files
            or not set(SDK_FILES)<=set(self.reviewed_sdk_files)):
            raise ValueError('Incomplete/changed SDK hashes')
        for name,h in self.reviewed_sdk_files.items():
            if name not in SDK_FILES or sha256(name)!=h:raise ValueError('Reviewed SDK changed: '+name)
        return True


def load_permit(path,arm):
    path=workspace_file(path);decision=read_json(path)
    permit=BoardPermit(path,sha256(path),decision.get('reviewed_files',{}),decision.get('reviewed_sdk_files',{}));permit.validate(arm)
    return permit


def require_confirmation(confirmation,permit,*,arm,label,previous_result_path):
    # A JSON record is evidence bookkeeping, not proof that a human power cycle
    # occurred. The operator may create this only after a new direct user reply.
    previous=workspace_file(previous_result_path)
    if (confirmation.get('source')!='human_user_current_session'
        or confirmation.get('physical_power_removed') is not True
        or confirmation.get('linux_boot_complete') is not True
        or not isinstance(confirmation.get('verbatim_reply'),str)
        or not confirmation['verbatim_reply'].strip()):
        raise ValueError('Fresh human physical cold-start confirmation required')
    if (confirmation.get('arm')!=arm or confirmation.get('label')!=label
        or confirmation.get('execution_record_sha256',confirmation.get('review_sha256'))!=permit.decision_sha256
        or confirmation.get('previous_result_sha256')!=sha256(previous)):
        raise ValueError('Confirmation belongs to another run/review/previous session')
    return previous


def release_request():
    return dict(schema='udw-e-board-release-v1',E_board_experiment_allowed=False,
        candidate_validation_accepted=False,closed_findings=[],allowed_arms=['S','H'],
        constraints=CONSTRAINTS,reviewed_files={name:sha256(ROOT/name) for name in CRITICAL_FILES},
        reviewed_sdk_files={name:sha256(name) for name in SDK_FILES},
        expected_device_versions=DEVICE_VERSIONS,
        note='Review request only; not an approval. No physical confirmation is generated.')
