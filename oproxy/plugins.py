"""
LLM Proxy Handler using proxy.py framework
Routes requests to appropriate LLM providers based on URL patterns
"""


import logging
# from typing import Dict, Any, Optional
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional
from urllib.parse import urlparse, urljoin, parse_qs

from proxy.http.parser import HttpParser
from proxy.http.url import Url
from proxy.http.server import ReverseProxyBasePlugin
from proxy.http.exception import HttpRequestRejected

from oproxy.config import get_provider_config, validate_provider_config, get_supported_providers, get_providers
from oproxy.utils import convert_url


class LLMProxyPlugin(ReverseProxyBasePlugin):
    """Proxy plugin for routing LLM requests to appropriate providers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def before_routing(self, request: HttpParser) -> Optional[HttpParser]:
        """Plugins can modify request, return response, close connection.
        If None is returned, request will be dropped and closed."""

        provider, _ = convert_url(request._url.remainder.decode())  # type: ignore

        # Validate provider configuration
        if not validate_provider_config(provider):
            self.logger.error(f"Invalid configuration for provider: {provider}")
            raise HttpRequestRejected(
                status_code=500,
                reason=f'Provider {provider} not properly configured'.encode()
            )

        # Get provider configuration
        provider_config = get_provider_config(provider)
        
        # Update request headers and URL
        self._update_request_headers(request, provider_config)

        return request  # pragma: no cover


    def routes(self) -> List[Union[str, Tuple[str, List[bytes]]]]:
        """Define URL patterns to match for routing"""
        router_patterns: List[Union[str, Tuple[str, List[bytes]]]] = ['/' + key + '/.*' for key in get_providers()]
        self.logger.debug(f"Router patterns: {router_patterns}")
        return router_patterns

    def handle_route(
        self,
        request: HttpParser,
        pattern: Any,
    ) -> Union[memoryview, Url]:
        """Implement this method if you have configured dynamic routes."""

        self.logger.debug(f"Request: {request.method} {request._url}")
        self.logger.debug(f"\tHeaders: {request.headers}")
        self.logger.debug(f"\tBody: {request.body}")

        provider, target_url = convert_url(request._url.remainder.decode())  # type: ignore

        # Get provider configuration
        provider_config = get_provider_config(provider)

        target_url = '/'.join([provider_config.get("base_url", ""), target_url])

        return  Url.from_bytes(target_url.encode())

    def _update_request_headers(self, request: HttpParser, provider_config: Dict[str, Any]):
        """Update request headers with provider-specific authentication"""
        headers = provider_config["headers"]
        
        if request.headers is None:
            request.headers = {}
        
        base_url = provider_config.get("base_url")
        if base_url:
            host = urlparse(base_url).netloc.encode()
            request.headers[b'host'] = (b'Host', host)
        
        # Add provider-specific headers
        for key, value in headers.items():
            header_key = key.encode()
            header_key_lower = key.lower().encode()
            header_value = value.format(**provider_config).encode()
            request.headers[header_key_lower] = (header_key, header_value)
