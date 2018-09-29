# Dockerfile that starts with a base dockerfile that has all dependencies
# pre-installed, and then runs all tests.
#
# To create a new base version, run the following command:
#
#   gcloud builds submit --config cloudbuild.yaml .
#

FROM gcr.io/google.com/tom-experiments/unrestricted-advex-base

# Copy in the new changes from this github commit
COPY . /usr/local/unrestricted-adversarial-examples/

# Run our tests
RUN find -name "*.pyc" -delete
RUN tox