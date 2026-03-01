Churn API
---

This project is meant to improve Churn Classification at our organization. It is meant for internal use ONLY! Currently, it uses a Random Forest Classifier and 10+ dependent variables to predict customer churn.


## How to Use

Right now, the API can only be ran locally; however, we expect to deploy it in production soon...

#### Installation
To use the API locally, first clone this repository. Then run

```bash
docker compose -f docker-compose.yml -f docker-compose.stage.yml up --build
```

You must have `docker-compose` installed.

Afterwards, you can check whether the API is ready by running the following command:
```bash
curl http://localhost/readiness
```

If the request does not return an error that means the API is ready to accept data.

#### Making Predictions
Predictions can be made on `/predict` and `/batch-predict` endpoints for a single data point and for multiple datapoints, respectively.

##### Example:
```bash
curl --header "Content-Type: application/json" \
   --request POST \
   --data '[{"customer_id":"CUST_00002","age":32,"gender":"Female","tenure_months":8,"monthly_charges":55.2,"total_charges":441.6,"contract_type":"Month-to-month","payment_method":"Electronic check","paperless_billing":"No","num_support_tickets":5,"num_logins_last_month":18,"feature_usage_score":3.2,"late_payments":2,"partner":"No","dependents":"No","internet_service":"DSL","online_security":"No","online_backup":"Yes","device_protection":"No","tech_support":"No","streaming_tv":"No","streaming_movies":"No","churn":1}, {"customer_id":"CUST_00003","age":28,"gender":"Male","tenure_months":3,"monthly_charges":89.95,"total_charges":269.85,"contract_type":"Month-to-month","payment_method":"Bank transfer","paperless_billing":"Yes","num_support_tickets":8,"num_logins_last_month":5,"feature_usage_score":2.1,"late_payments":3,"partner":"No","dependents":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"Yes","streaming_movies":"Yes","churn":1}]' \
   http://localhost/batch-predict
```

Notice that when requesting on `/predict` the body must be of the form `{data}` as opposed to `[{data1}, {data2}, ...]`.

The response is a JSON with a list of `prediction` and `probability` associated to each prediction.

For more info follow the http://localhost/docs which has useful information on each endpoint.

![Screenshot of OpenAPI documentation.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%206.png)

## How to Contribute
> [!IMPORTANT]
> If you want to contribute, or have been assigned to maintain this repository you must understand the current CI/CD in this project.

Churn-api uses Python 3.12.

Because the model and the current amount of data is light the training is performed locally during docker compose and on GitHub Actions.

### CI
The project has GitHub Actions Workflows that can be located in `.github/workflows/ci.yml`

Those actions are:

* Setting up Python
* Installing dependencies from `requirements.txt`
* Training the model (currently using `scikit-learn`).
* Checking for linting errors using `flake8`.
* Checking the typehinting using `mypy`
* Running tests using `pytest` and `pytest-cov`.
    * All the tests are located in `/tests`
    * There are two test files now:

        1. `test_api.py` for testing the API endpoints.
        2. `test_prediction.py` for testing the model.
* Running safety scans using `safety`.
    * The repo has the api key saved.

It must be noted that the CI workflow currently only runs for pushes and PRs on the `main` branch.

Here are some examples of the Actions working after a push.
![Screenshot of GHA1.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%201.png)

![Screenshot of GHA2.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%202.png)

Using the GitHub Actions extension in VSCode those actions can be inspected directly in the IDE.
![Screenshot VSCode GHA.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%203.png)

### CD
There are three so-called "environments": development, staging, and production and they are separeted only by their respective docker-compose files:

* Development: `docker-compose.dev.yml`
* Staging: `docker-compose.stage.yml`
* Production: `docker-compose.prod.yml`

#### Production
The future remote server will run the following command:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

It is connected to the Digital Ocean Container Registry: `registry.digitalocean.com/churn-api`. Where each build is tagged with a version, like this: `churn-api:v1.0`.

![Screenshot of DOCR.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%207.png)

Whenever a new build has to be rolled out first make sure to have `doctl` installed and authenticated (for the access key ask an adult).

Build the image locally
```bash
docker build -t churn-api:v[<version>] .
```

Tag the image
```bash
docker tag churn-api:v[<version>] registry.digitalocean.com/churn-api/churn-api:v[<version>]
```

Push the image
```bash
docker push registry.digitalocean.com/churn-api/churn-api:v[<version>]
```

Now, the new image should be saved in the DOCR. Running the docker compose in production will use the specified version.


![Screenshot of running the container in production for the first time.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%208.png)

Since the models do not require a lot of compute to train a specialized GPU service will ___not___ be used.

#### Staging
This is meant to be ran locally when the developer wants to make sure that his code will work in production. It is currently as close as possible to the production compose config.

#### Development
To be able to develop VSCode is recommended with the following extensions installed:

* GitHub Actions.
* Docker Containers.
* Python extensions.

A Python installation is required, preferrably version 3.12.

In the repository, create the virtual environment and install the requirements.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> [!IMPORTANT]
> All the pip modules must be pinned to a version `useful-package==9.9.99`.

The `requirements.txt` has `debugpy` which works well with the VSCode debugger. The project already has the necessary configuration for that to work in `.vscode/launch.json`. Only the development image can be attached to.

Before attaching the debugger an image instance must be running.

![Screenshot of the debugger.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%204.png)

The debugger is useful because breakpoints during runtime can be set, and the environment including variables can be inspected whenever a breakpoint is hit.

Here's an example where a breakpoint was set within the readiness endpoint and hit after a request was made using `curl` command.

![Screenshot of running the container in production for the first time.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%205.png)

Additionally, thanks to FastAPI, document is automatically generated as long as the models in `src/models.py` are correctly type annotated. Same goes for the API endpoints in `src/main.py`. Overall, all the parameters should be typehinted and given an example especially whenever the Python's typehints are ambiguous.

To run the tests use the following command:
```bash
pytest --cov=src
```

This command will run coverage tests on the `src` folder.

Here's an example:

![Screenshot of running the container in production for the first time.](https://raw.githubusercontent.com/polar0n/churn-api/refs/heads/main/images/Screenshot%209.png)

These tests are also performed after pushes to the main branch.

#### Further Notes
Whenever an issue branch is created, it should be max three words, ex. `01-fix-god-model`. Commits withing issues should end with `#01` (notice the issue number).

Currently, this API is for internal use within the organization, because of that I do not expect it to have a lot of requests and therefore, a load-balancer such as Nginx is completely unnecessary.