# Technical Assignment
E2E Model Generation

## Requirements
- Docker
- Docker-compose
  
## Setup (webserver)

1. `cd webserver && docker-compose up`
2. For using Swagger UI go to `http://localhost:8000/`, otherwise hit `http://127.0.0.1:8000/house_pricing/predict` endpoint with query parameters:
    - address
    - type
    - bedrooms
    - latitude
    - longitude
    - area
    - tenure
    - is_newbuild
    - date

   Example:
   ```bash
   curl --request GET 'http://127.0.0.1:8000/house_pricing/predict/?address=Flat%2029,%20Mulberry%20Court,%201,%20School%20Mews,%20London,%20Greater%20London%20E1%200EW&type=Flat%20&bedrooms=19&latitude=51.43061&longitude=-0.08388&area=SE21&tenure=Freehold&is_newbuild=0&date=2019-11-11%2000:00:00+00:00'
   ```
    Result:
    ```json
    {
        "result": {
            "estimated_price": 745428
        }
    }
