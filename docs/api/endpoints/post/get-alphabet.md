[back](../../index.md)
# POST - Get Alphabet
**URL:** `{root_url}/get-alphabet`

## Description

Gets an image and return the letter it represent in LSF (French Sign Language) or nothing if it's not a letter.

## Request

### Header
You must set in your request headers the following values:

```json
{
    "Content-Type": "multipart/form-data"
}
```

### Body
The request body should be in the **form-data** format and include the following parameters:

| Key  | Type   | Description                                              |
|------|--------|----------------------------------------------------------|
| file | File   | The image file to be processed. (in .jpg or .png format) |

## Response
### 200
```json
{
    "message": "A", // A letter in uppercase or "null" if no letter is recognized.
}
```
### 400
> When the body or its values are incorrect.
```json
{
    "message": "Error message"
}
```
### 401
> Response defined in [index.md](../../index.md)

## Example
```sh
curl -X POST "{root_url}/get-alphabet" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@B.jpg"
```
