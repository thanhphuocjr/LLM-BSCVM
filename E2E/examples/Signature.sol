pragma solidity ^0.8.20;

contract VulnerableVault {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    receive() external payable {}

    function withdrawWithSignature(
        address payable to,
        uint256 amount,
        bytes memory signature
    ) external {
        bytes32 messageHash = keccak256(
            abi.encodePacked(to, amount)
        );

        bytes32 ethSignedMessageHash = keccak256(
            abi.encodePacked(
                "\x19Ethereum Signed Message:\n32",
                messageHash
            )
        );

        address signer = recoverSigner(
            ethSignedMessageHash,
            signature
        );

        require(signer == owner, "Invalid signature");

        (bool success, ) = to.call{value: amount}("");
        require(success, "Transfer failed");
    }

    function recoverSigner(
        bytes32 hash,
        bytes memory signature
    ) internal pure returns (address) {
        require(signature.length == 65, "Invalid signature");

        bytes32 r;
        bytes32 s;
        uint8 v;

        assembly {
            r := mload(add(signature, 32))
            s := mload(add(signature, 64))
            v := byte(0, mload(add(signature, 96)))
        }

        return ecrecover(hash, v, r, s);
    }
}