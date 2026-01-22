# SPDX-License-Identifier: Apache-2.0
import requests


def get_instance_type():
    "Gets the current instance type using an IMDSv2 token"
    try:
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        )
        token = token_response.text

        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        )
        print(f"detected instance type: {response.text}")
        return response.text
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve instance type from EC2 metadata service: {e}"
        ) from e
