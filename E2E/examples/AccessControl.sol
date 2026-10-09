// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VulnerableToken {
    mapping(address => uint256) public balances;

    address public owner;
    uint256 public totalSupply;

    constructor() {
        owner = msg.sender;
    }

    function mint(address to, uint256 amount) external {
        // Vulnerability:
        // Không kiểm tra quyền của người gọi
        balances[to] += amount;
        totalSupply += amount;
    }

    function transfer(address to, uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}