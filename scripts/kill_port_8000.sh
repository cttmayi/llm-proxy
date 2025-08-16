#!/bin/bash

lsof -i :8000

lsof -ti:8000 | xargs kill -9