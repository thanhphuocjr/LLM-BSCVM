#!/usr/bin/env python3
"""
Sinh ~1000 mẫu SYNTHETIC dạng HARD-NEGATIVE cho phân loại nhị phân.

Nguyên tắc: với mỗi lỗ hổng, sinh một CẶP gần-giống-hệt:
  - Vulnerable: chứa đúng anti-pattern (reentrancy, thiếu SafeMath, thiếu onlyOwner, tx.origin, ...)
  - Safe: bản ĐÃ SỬA của chính nó (CEI / SafeMath / onlyOwner / msg.sender / check-return, ...)
=> Cặp chỉ khác nhau ở phần vá lỗi -> buộc model học pattern THẬT, không học từ khoá bề mặt.

Đa dạng hoá để KHÔNG tạo shortcut mới:
  - Solidity version trải cả 0.4.x -> 0.8.x cho CẢ hai nhãn (phá "version cũ = vuln")
  - Nhiều theme (Vault/Token/Crowdsale/...) + tên biến/hàm ngẫu nhiên
  - Độ dài thay đổi bằng filler ERC20-ish (vuln và safe cùng mức filler -> length không tách nhãn)

Output: synthetic_samples.jsonl  (schema: code,label,source,categories,swc_ids,granularity,fix_type)
"""
import json, random, re, hashlib

SEED = 7
rng = random.Random(SEED)

THEMES = ["Vault", "Bank", "TokenSale", "Crowdsale", "Staking", "Auction", "Escrow",
          "Wallet", "Presale", "Vesting", "Lottery", "Treasury", "Dividend",
          "Marketplace", "Farm", "Pool", "Exchange", "Fund", "Reserve", "Deposit"]
VERSIONS = ["^0.4.24", "0.4.25", "^0.5.0", "0.5.16", "^0.6.0", "0.6.12",
            "0.7.6", "^0.8.0", "0.8.10", "0.8.17", "0.8.19"]

def minor_of(v):
    m = re.search(r"0\.(\d+)", v); return int(m.group(1)) if m else 8
def is08(v): return minor_of(v) >= 8

SAFEMATH = """library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) { uint256 c = a + b; require(c >= a, "add overflow"); return c; }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) { require(b <= a, "sub underflow"); return a - b; }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) { if (a == 0) return 0; uint256 c = a * b; require(c / a == b, "mul overflow"); return c; }
    function div(uint256 a, uint256 b) internal pure returns (uint256) { require(b > 0, "div zero"); return a / b; }
}"""

def owned(v):
    ctor = "constructor() public { owner = msg.sender; }" if minor_of(v) < 7 else "constructor() { owner = msg.sender; }"
    return ("contract Owned {\n    address public owner;\n    " + ctor +
            "\n    modifier onlyOwner() { require(msg.sender == owner, \"not owner\"); _; }\n}")

def pay_type(v): return "address payable" if minor_of(v) >= 5 else "address"
def pay_cast(x, v):
    m = minor_of(v)
    if m >= 6: return f"payable({x})"
    if m == 5: return f"address(uint160({x}))"
    return x
def send_value(target, amount, v, checked):
    """Sinh câu lệnh gửi ETH; checked=True -> có require(return)."""
    m = minor_of(v)
    if m >= 6:
        if checked: return f'(bool _ok, ) = {target}.call{{value: {amount}}}(""); require(_ok, "send failed");'
        return f'{target}.call{{value: {amount}}}("");'
    if m == 5:
        if checked: return f'(bool _ok, ) = {target}.call.value({amount})(""); require(_ok, "send failed");'
        return f'{target}.call.value({amount})("");'
    if checked: return f'require({target}.call.value({amount})(), "send failed");'
    return f'{target}.call.value({amount})();'

def ctor(v, extra=""):
    vis = " public" if minor_of(v) < 7 else ""
    return f"constructor(){vis} {{ owner = msg.sender;{extra} }}"

# ---------------- filler để đa dạng độ dài (ERC20-ish, vô hại) ----------------
FILLERS = [
    """    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    mapping(address => mapping(address => uint256)) public allowance;
    function approve(address spender, uint256 value) public returns (bool) { allowance[msg.sender][spender] = value; emit Approval(msg.sender, spender, value); return true; }
    function balanceOf(address who) public view returns (uint256) { return balances[who]; }""",
    """    string public symbol;
    uint8 public decimals = 18;
    bool public paused;
    function pause() public onlyOwner { paused = true; }
    function unpause() public onlyOwner { paused = false; }
    function setSymbol(string memory s) public onlyOwner { symbol = s; }""",
    """    uint256 public cap;
    mapping(address => bool) public whitelist;
    function setCap(uint256 c) public onlyOwner { cap = c; }
    function addWhitelist(address a) public onlyOwner { whitelist[a] = true; }
    function isWhitelisted(address a) public view returns (bool) { return whitelist[a]; }""",
    """    uint256 public createdAt;
    address public feeCollector;
    uint256 public feeBps = 30;
    function setFee(uint256 bps) public onlyOwner { require(bps <= 1000); feeBps = bps; }
    function setCollector(address c) public onlyOwner { feeCollector = c; }""",
]
def filler(n):
    fs = FILLERS[:]; rng.shuffle(fs)
    return "\n".join(fs[:n])

# ---------------- GENERATORS: mỗi cái trả (category, swc, core_vuln, core_safe, fix_type, need_sm_safe) ----------------
def g_reentrancy(v):
    amt = "amount"
    vuln = f"""    mapping(address => uint256) private shares;
    function deposit() public payable {{ shares[msg.sender] += msg.value; }}
    function withdraw(uint256 {amt}) public {{
        require(shares[msg.sender] >= {amt}, "insufficient");
        {send_value("msg.sender", amt, v, True)}
        shares[msg.sender] -= {amt};   // state cap nhat SAU external call -> reentrancy
    }}"""
    if rng.random() < 0.5:  # fix = checks-effects-interactions
        safe = f"""    mapping(address => uint256) private shares;
    function deposit() public payable {{ shares[msg.sender] += msg.value; }}
    function withdraw(uint256 {amt}) public {{
        require(shares[msg.sender] >= {amt}, "insufficient");
        shares[msg.sender] -= {amt};   // effects TRUOC (checks-effects-interactions)
        {send_value("msg.sender", amt, v, True)}
    }}"""
        fix = "checks_effects_interactions"
    else:  # fix = reentrancy guard
        safe = f"""    mapping(address => uint256) private shares;
    bool private locked;
    modifier noReentry() {{ require(!locked, "reentrant"); locked = true; _; locked = false; }}
    function deposit() public payable {{ shares[msg.sender] += msg.value; }}
    function withdraw(uint256 {amt}) public noReentry {{
        require(shares[msg.sender] >= {amt}, "insufficient");
        {send_value("msg.sender", amt, v, True)}
        shares[msg.sender] -= {amt};
    }}"""
        fix = "reentrancy_guard"
    return ("reentrancy", "SWC-107", vuln, safe, fix, False)

def g_arithmetic(v):
    if is08(v):  # 0.8 checked -> vuln phai dung unchecked
        vuln = """    function transfer(address to, uint256 value) public returns (bool) {
        unchecked { balances[msg.sender] -= value; balances[to] += value; }  // unchecked -> overflow/underflow
        emit Transfer(msg.sender, to, value); return true;
    }"""
        safe = """    function transfer(address to, uint256 value) public returns (bool) {
        require(balances[msg.sender] >= value, "insufficient");
        balances[msg.sender] -= value; balances[to] += value;   // 0.8 checked arithmetic
        emit Transfer(msg.sender, to, value); return true;
    }"""
        return ("arithmetic", "SWC-101", vuln, safe, "checked_math_08", False)
    else:  # pre-0.8: vuln thieu SafeMath, safe dung SafeMath
        vuln = """    function transfer(address to, uint256 value) public returns (bool) {
        balances[msg.sender] -= value;   // khong kiem tra -> underflow
        balances[to] += value;           // overflow
        emit Transfer(msg.sender, to, value); return true;
    }"""
        safe = """    function transfer(address to, uint256 value) public returns (bool) {
        require(balances[msg.sender] >= value, "insufficient");
        balances[msg.sender] = balances[msg.sender].sub(value);
        balances[to] = balances[to].add(value);
        emit Transfer(msg.sender, to, value); return true;
    }"""
        return ("arithmetic", "SWC-101", vuln, safe, "safemath", True)

def g_access(v):
    fn = rng.choice(["mint", "setRate", "withdrawAll", "setOwner"])
    if fn == "mint":
        body = "totalSupply += amount; balances[to] += amount; emit Transfer(address(0), to, amount);"
        sig = "mint(address to, uint256 amount)"
    elif fn == "setRate":
        body = "rate = newRate;"; sig = "setRate(uint256 newRate)"
    elif fn == "withdrawAll":
        body = send_value(pay_cast("msg.sender", v), "address(this).balance", v, True); sig = "withdrawAll()"
    else:
        body = "owner = newOwner;"; sig = "setOwner(address newOwner)"
    extra = "    uint256 public rate;\n" if fn == "setRate" else ""
    vuln = f"{extra}    function {sig} public {{ {body} }}   // THIEU onlyOwner"
    safe = f"{extra}    function {sig} public onlyOwner {{ {body} }}"
    return ("access_control", "SWC-105", vuln, safe, "add_onlyowner", False)

def g_txorigin(v):
    vuln = f"""    function transferTo({pay_type(v)} to, uint256 amount) public {{
        require(tx.origin == owner, "not owner");   // tx.origin -> phishing
        {send_value("to", "amount", v, True)}
    }}"""
    safe = f"""    function transferTo({pay_type(v)} to, uint256 amount) public {{
        require(msg.sender == owner, "not owner");   // dung msg.sender
        {send_value("to", "amount", v, True)}
    }}"""
    return ("access_control", "SWC-115", vuln, safe, "msg_sender", False)

def g_unchecked_call(v):
    vuln = f"""    function payout({pay_type(v)} to, uint256 amount) public onlyOwner {{
        {send_value("to", "amount", v, False)}   // KHONG kiem tra return -> silent fail
    }}"""
    safe = f"""    function payout({pay_type(v)} to, uint256 amount) public onlyOwner {{
        {send_value("to", "amount", v, True)}
    }}"""
    return ("unchecked_low_calls", "SWC-104", vuln, safe, "check_return", False)

def g_selfdestruct(v):
    vuln = f"    function kill() public {{ selfdestruct({pay_cast('msg.sender', v)}); }}   // ai cung goi duoc"
    safe = f"    function kill() public onlyOwner {{ selfdestruct({pay_cast('owner', v)}); }}"
    return ("access_control", "SWC-106", vuln, safe, "protect_selfdestruct", False)

def g_randomness(v):
    vuln = """    address[] public players;
    address public winner;
    function enter() public payable { require(msg.value >= 0.1 ether); players.push(msg.sender); }
    function draw() public onlyOwner {
        uint256 idx = uint256(keccak256(abi.encodePacked(block.timestamp, block.difficulty, msg.sender))) % players.length;
        winner = players[idx];   // rand tu block -> miner thao tung duoc
    }"""
    safe = """    address[] public players;
    address public winner;
    bytes32 private commit;
    function enter() public payable { require(msg.value >= 0.1 ether); players.push(msg.sender); }
    function commitSeed(bytes32 h) public onlyOwner { commit = h; }   // commit-reveal
    function draw(uint256 seed) public onlyOwner {
        require(keccak256(abi.encodePacked(seed)) == commit, "bad seed");
        winner = players[seed % players.length];
    }"""
    return ("bad_randomness", "SWC-120", vuln, safe, "commit_reveal", False)

def g_timestamp(v):
    vuln = """    uint256 public deadline;
    function claimBonus() public {
        require(block.timestamp % 15 == 0, "not lucky");   // phu thuoc timestamp chinh xac -> miner nudge
        balances[msg.sender] += 100 ether;
    }"""
    safe = """    uint256 public startBlock;
    uint256 public constant WINDOW = 6000;
    function claimBonus() public {
        require(block.number >= startBlock && block.number < startBlock + WINDOW, "closed");  // dung block.number, cua so rong
        balances[msg.sender] += 100 ether;
    }"""
    return ("time_manipulation", "SWC-116", vuln, safe, "block_number", False)

def g_dos(v):
    vuln = f"""    {pay_type(v)}[] public investors;
    mapping(address => uint256) public dividends;
    function distribute() public onlyOwner {{
        for (uint256 i = 0; i < investors.length; i++) {{
            investors[i].transfer(dividends[investors[i]]);   // 1 revert chan tat ca; gas khong gioi han
        }}
    }}"""
    safe = f"""    {pay_type(v)}[] public investors;
    mapping(address => uint256) public dividends;
    function withdrawDividend() public {{                      // pull-over-push
        uint256 amt = dividends[msg.sender];
        require(amt > 0, "nothing");
        dividends[msg.sender] = 0;
        {send_value("msg.sender", "amt", v, True)}
    }}"""
    return ("denial_service", "SWC-113", vuln, safe, "pull_payment", False)

def g_delegatecall(v):
    vuln = """    function execute(address target, bytes memory data) public {
        (bool ok, ) = target.delegatecall(data);   // delegatecall toi dia chi tuy y -> chiem quyen
        require(ok, "call failed");
    }"""
    safe = """    address public immutable logic;
    function execute(bytes memory data) public onlyOwner {
        (bool ok, ) = logic.delegatecall(data);   // chi delegatecall toi logic co dinh, onlyOwner
        require(ok, "call failed");
    }"""
    if minor_of(v) < 6:  # immutable can 0.6+
        safe = safe.replace("address public immutable logic;", "address public logic;")
    return ("access_control", "SWC-112", vuln, safe, "restrict_delegatecall", False)

def g_frontrun_approve(v):
    vuln = """    mapping(address => mapping(address => uint256)) public allowed;
    function approve(address spender, uint256 value) public returns (bool) {
        allowed[msg.sender][spender] = value;   // race: spender co the tieu ca cu lan moi
        return true;
    }"""
    safe = """    mapping(address => mapping(address => uint256)) public allowed;
    function increaseAllowance(address spender, uint256 adder) public returns (bool) {
        allowed[msg.sender][spender] += adder;   // tang/giam thay vi set -> tranh race
        return true;
    }
    function decreaseAllowance(address spender, uint256 subber) public returns (bool) {
        uint256 cur = allowed[msg.sender][spender];
        allowed[msg.sender][spender] = subber > cur ? 0 : cur - subber;
        return true;
    }"""
    return ("front_running", "SWC-114", vuln, safe, "increase_allowance", False)

GENERATORS = [g_reentrancy, g_arithmetic, g_access, g_txorigin, g_unchecked_call,
              g_selfdestruct, g_randomness, g_timestamp, g_dos, g_delegatecall, g_frontrun_approve]

# ---------------- lắp ráp contract hoàn chỉnh ----------------
def strip_giveaway(code):
    """Xoá MỌI comment // ... để mẫu vuln không tự tố lỗ hổng (chống leak qua comment).
    Không đụng phép chia a / b (chỉ khớp '//')."""
    code = re.sub(r"[ \t]*//[^\n]*", "", code)         # trailing/line // comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)  # block comments (nếu có)
    code = re.sub(r"[ \t]+\n", "\n", code)             # xoá khoảng trắng cuối dòng
    code = re.sub(r"\n{3,}", "\n\n", code)             # gộp dòng trống thừa
    return code

def assemble(theme, v, core, label, uses_safemath, n_filler):
    parts = [f"pragma solidity {v};"]
    if uses_safemath:
        parts.append(SAFEMATH)
    parts.append(owned(v))
    using = "\n    using SafeMath for uint256;" if uses_safemath else ""
    head = (f"contract {theme} is Owned {{{using}\n"
            f"    string public name = \"{theme}\";\n"
            f"    uint256 public totalSupply;\n"
            f"    mapping(address => uint256) public balances;\n"
            f"    event Transfer(address indexed from, address indexed to, uint256 value);\n")
    fill = filler(n_filler)
    body = head + core + "\n" + (fill + "\n" if fill else "") + "}"
    parts.append(body)
    return strip_giveaway("\n\n".join(parts))

def build(n_pairs_per_gen=48):
    out, seen = [], set()
    def sig(code):
        s = re.sub(r"\b[A-Za-z_]\w*\b", "X", code); return hashlib.md5(re.sub(r"\s+", "", s).encode()).hexdigest()
    for gen in GENERATORS:
        made = 0; tries = 0
        while made < n_pairs_per_gen and tries < n_pairs_per_gen * 6:
            tries += 1
            v = rng.choice(VERSIONS); theme = rng.choice(THEMES); nf = rng.randint(0, 3)
            cat, swc, cv, cs, fix, need_sm = gen(v)
            # vuln arithmetic pre-0.8 dùng ops thô, không cần using; safe cần SafeMath
            code_v = assemble(theme, v, cv, "Vulnerable", uses_safemath=False, n_filler=nf)
            code_s = assemble(theme, v, cs, "Safe", uses_safemath=need_sm, n_filler=nf)
            kv, ks = sig(code_v), sig(code_s)
            if kv in seen or ks in seen or kv == ks:
                continue
            seen.add(kv); seen.add(ks)
            base = {"source": "synthetic_hardneg", "categories": [cat], "swc_ids": [swc],
                    "granularity": "contract"}
            out.append({**base, "code": code_v, "label": "Vulnerable", "fix_type": ""})
            out.append({**base, "code": code_s, "label": "Safe", "fix_type": fix})
            made += 1
    rng.shuffle(out)
    return out

if __name__ == "__main__":
    rows = build()
    with open("synthetic_samples.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections, statistics as st
    print(f"Sinh {len(rows)} mẫu | nhãn {dict(collections.Counter(r['label'] for r in rows))}")
    print("theo category:", dict(collections.Counter(r['categories'][0] for r in rows)))
    Lv = [len(r['code']) for r in rows if r['label'] == 'Vulnerable']
    Ls = [len(r['code']) for r in rows if r['label'] == 'Safe']
    print(f"len median Vuln={int(st.median(Lv))} Safe={int(st.median(Ls))}")
    def mn(v): return re.search(r'0\.(\d+)', v).group(0)
    vv = collections.Counter(mn(re.search(r'pragma solidity[^;]*', r['code']).group(0)) for r in rows if r['label']=='Vulnerable')
    vs = collections.Counter(mn(re.search(r'pragma solidity[^;]*', r['code']).group(0)) for r in rows if r['label']=='Safe')
    print("version Vuln:", dict(vv)); print("version Safe:", dict(vs))
    print("Ghi synthetic_samples.jsonl")
