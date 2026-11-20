#!/usr/bin/env bash
set -o errexit

apt-get update
apt-get install -y espeak libespeak1

pip install -r requirements.txt gunicorn
