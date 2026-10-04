import pytest
from inference.serve import BaselinePolicyServer
from paw_policy_toolkit.policy_service import ConfigureRequest


class TestDeferredConfigure:
    def test_constructor_does_not_read_checkpoint(self, tmp_path):
        server = BaselinePolicyServer(tmp_path / "missing", "cpu", "model")
        with pytest.raises(RuntimeError, match="Configure"):
            server.activate()
        server.close()

    def test_load_once_reuses_across_tasks_and_episodes(self, monkeypatch, tmp_path):
        calls = []

        class Policy:
            def reset(self):
                calls.append("reset")

        def load(server):
            calls.append("load")
            server._policy = Policy()
            server._cameras = ()

        monkeypatch.setattr(BaselinePolicyServer, "_load", load)
        server = BaselinePolicyServer(tmp_path, "cpu", "model")
        for task in ("pick", "place", "pick"):
            server.do_configure(
                ConfigureRequest(model_id="model", task_instruction=task)
            )
            server.activate()
            assert server._task_instruction == task
        assert calls == ["load", "reset", "reset", "reset"]
        server.close()

    def test_failed_initialization_blocks_run_and_retry(self, monkeypatch, tmp_path):
        calls = []

        def load(server):
            calls.append("load")
            raise RuntimeError("warmup failed")

        monkeypatch.setattr(BaselinePolicyServer, "_load", load)
        server = BaselinePolicyServer(tmp_path, "cpu", "model")
        request = ConfigureRequest(model_id="model", task_instruction="pick")
        with pytest.raises(RuntimeError, match="warmup failed"):
            server.do_configure(request)
        with pytest.raises(RuntimeError, match="Configure"):
            server.activate()
        with pytest.raises(RuntimeError, match="initialization failed"):
            server.do_configure(request)
        assert calls == ["load"]
        server.close()


@pytest.mark.parametrize("fail", [False, True])
def test_grpc_is_ready_before_loading_and_gates_run(monkeypatch, tmp_path, fail):
    import socket
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import grpc
    from paw_policy_toolkit.policy_service import PolicyServiceStub
    from paw_policy_toolkit.server import serve

    loading = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    loads = []

    def load(server):
        loads.append(1)
        loading.set()
        assert release.wait(10)
        if fail:
            raise RuntimeError("warmup failed")
        server._policy = SimpleNamespace(reset=lambda: None)
        server._cameras = ()

    monkeypatch.setattr(BaselinePolicyServer, "_load", load)
    policy = BaselinePolicyServer(tmp_path / "absent", "cpu", "model")
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    thread = threading.Thread(
        target=serve,
        args=(policy,),
        kwargs={
            "host": "localhost",
            "port": port,
            "tls": False,
            "verifier": None,
            "stop_event": stopped,
        },
    )
    thread.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = PolicyServiceStub(channel)
            assert loads == []
            with pytest.raises(grpc.RpcError, match="successful Configure"):
                list(stub.Run(iter(()), timeout=5))
            with ThreadPoolExecutor(max_workers=1) as pool:
                configured = pool.submit(
                    stub.Configure,
                    ConfigureRequest(model_id="model", task_instruction="pick"),
                    timeout=10,
                )
                assert loading.wait(5)
                with pytest.raises(grpc.RpcError) as error:
                    list(stub.Run(iter(()), timeout=5))
                assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                release.set()
                if fail:
                    with pytest.raises(grpc.RpcError, match="warmup failed"):
                        configured.result()
                    with pytest.raises(grpc.RpcError, match="successful Configure"):
                        list(stub.Run(iter(()), timeout=5))
                else:
                    configured.result()
                    for task in ("place", "pick"):
                        stub.Configure(
                            ConfigureRequest(model_id="model", task_instruction=task),
                            timeout=5,
                        )
                        assert list(stub.Run(iter(()), timeout=5)) == []
                        assert policy._task_instruction == task
            assert loads == [1]
    finally:
        release.set()
        stopped.set()
        thread.join(10)
        policy.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("error_type", ["RuntimeError", "torch.OutOfMemoryError"])
def test_main_exits_after_initialization_failure(tmp_path, error_type):
    import socket
    import subprocess
    import sys

    import grpc
    from paw_policy_toolkit.policy_service import PolicyServiceStub

    (tmp_path / "config.json").write_text("{}")
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    script = f"""
import torch
from inference.serve import BaselinePolicyServer, main

def fail_load(self):
    raise {error_type}("initialization failed for test")

BaselinePolicyServer._load = fail_load
main()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            "--checkpoint",
            str(tmp_path),
            "--model-id",
            "model",
            "--device",
            "cpu",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=15)
            stub = PolicyServiceStub(channel)
            for request in (
                ConfigureRequest(model_id="wrong", task_instruction="pick"),
                ConfigureRequest(model_id="model", task_instruction=""),
            ):
                with pytest.raises(grpc.RpcError) as error:
                    stub.Configure(request, timeout=5)
                assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                assert process.poll() is None
            with pytest.raises(grpc.RpcError) as error:
                stub.Configure(
                    ConfigureRequest(model_id="model", task_instruction="pick"),
                    timeout=5,
                )
            assert "initialization failed for test" in error.value.details()
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 1, output
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_main_rejects_missing_checkpoint_before_serving(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference.serve",
            "--checkpoint",
            str(tmp_path),
            "--device",
            "cpu",
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "no config.json" in result.stderr
    assert "Policy server listening" not in result.stderr
