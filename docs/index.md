# Trio-Signo AI Backend API

The following document will describe thoroughly how to use the API to communicate
with the AI back-end part of Trio-Signo.

# Information

This API follow the REST convention.

Every request and response are in JSON format. No other format is supported yet.

# Header and Authentication

Except for [register](endpoints/post/register.md) and [log in](endpoints/post/login.md)
endpoints that are used to aquire a **token**, the field `Authorization` is expected
**in the header of any api call** in the following format:
```json
{
    "Authorization": "Bearer {your_token}",
}
```

## Response

If the token is missing or invalid, the response will be a 401 with the following message:
```
Unauthorized
```

# API Calls

## POST
- [Get Alphabet](endpoints/post/get_alphabet.md)
- [Log in](endpoints/post/login.md)
- [Create AREA](endpoints/post/create_area.md)
