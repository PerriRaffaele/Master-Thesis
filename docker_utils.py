import os, uuid, subprocess

"""
Docker utility functions
"""


def start_docker_container(container_id: str, container_image: str) -> None:
    """
    Start a docker container for the given image and assign the given id.

    :param container_id: The container id
    :param container_image: The container image
    """
    result = subprocess.run(f'docker run --name {container_id} --entrypoint tail -d {container_image} -f /dev/null',
                            shell=True)
    if result.returncode != 0:
        raise Exception(f"Error creating the container {container_image}")

    result = subprocess.run(f'docker exec {container_id} ls /app', shell=True)
    if result.returncode != 0:
        print(f"Directory /app does not exist in the container {container_id}, creating it...")
        result = subprocess.run(f'docker exec {container_id} mkdir -p /app', shell=True)
        if result.returncode != 0:
            raise Exception(f"Error creating the /app directory in the container {container_id}")


def remove_docker_container(container_id: str) -> None:
    """
    Remove the docker container with the given id.

    :param container_id: The container id
    """

    result = subprocess.run(f'docker rm -f {container_id}', shell=True)
    if result.returncode != 0:
        raise Exception(f"Error removing the container {container_id}")


def copy_code(code: str, container_id: str) -> str:
    """
    Create a copy of the script in the container. Returns the path to the copied script in the container.

    :param code: The content of the source code to copy
    :param container_id: The container id where the code will be copied
    :return: The path of the copied script in the container
    """
    temp_dir = "temp"
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    filename = f"temp_script_{uuid.uuid4()}.py"
    store_filepath = f"{temp_dir}/{filename}"

    container_path = f"/app/{filename}"
    with open(store_filepath, 'w') as f:
        f.write(code)
        f.flush()
        result = subprocess.run(f'docker cp {store_filepath} {container_id}:{container_path}', shell=True)
        if result.returncode != 0:
            raise Exception("Error copying the file to the container")

    os.remove(store_filepath)
    return container_path


def copy_file(src_filepath: str, trg_filepath: str, container_id: str) -> str:
    """
    Copy a file from the host to the container.
    """
    result = subprocess.run(f'docker cp {src_filepath} {container_id}:{trg_filepath}', shell=True)
    if result.returncode != 0:
        raise Exception("Error copying the file to the container")


def eval_script(container_id: str, command: str, path: str) -> tuple:
    """
    Implementation adapted from eval_r.py from the MultiPL-E project (https://github.com/nuprl/MultiPL-E)

    :param container_id: The container id where the script is located
    :param command: The command to run the script inside the container
    :param path: The path inside the container where the script is located
    :return: A tuple containing the return code and the output message
    """
    return_code = -1
    output_message = ""
    try:
        output = subprocess.run(['docker', 'exec', container_id, command, path], capture_output=True, timeout=240)
        return_code = output.returncode
        if output.returncode != 0:
            output_message += "Exception:\n"
        output_message += output.stdout.decode('utf-8') if output.stdout else ""
        output_message += output.stderr.decode('utf-8') if output.stderr else ""
    except subprocess.TimeoutExpired as exc:
        output_message += "Timeout during the execution of the test suite.\n"
        output_message += exc.stdout.decode('utf-8') if exc.stdout else ""
        output_message += exc.stderr.decode('utf-8') if exc.stderr else ""
    except subprocess.CalledProcessError as exc:
        output_message += "Error:\n"
        output_message += exc.stdout.decode('utf-8') if exc.stdout else ""
        output_message += exc.stderr.decode('utf-8') if exc.stderr else ""

    # Remove the file after running the script
    subprocess.run(['docker', 'exec', container_id, 'rm', '-f', path])
    return return_code, output_message