# System Stress Test Report (Phase 128)
Date: 2026-01-12 20:46:22

## Category: CONTACT
| Query | Status | Latency (s) | Notes |
|---|---|---|---|
| เบอร์ ชจญ. | SLOW | 194.09 |  |
| เบอร์ ภูเก็ต | SLOW | 48.63 |  |
| เบอร์ งาน smc | SLOW | 22.30 |  |
| ติดต่อคุณสมชาย | SLOW | 48.15 |  |
| เบอร์ ip network | SLOW | 46.20 |  |
| เบอร์ ผส.พส. | SLOW | 48.07 |  |
| เบอร์ noc | SLOW | 56.59 |  |
| เบอร์ แผนกจัดซื้อ | SLOW | 35.44 |  |
| เบอร์ คุณวิชัย | SLOW | 216.42 |  |
| ติดต่อ helpdesk | SLOW | 37.16 |  |

## Category: PERSON
| Query | Status | Latency (s) | Notes |
|---|---|---|---|
| คุณสมชาย | SLOW | 43.93 |  |
| ผจก ภ.4 คือใคร | SLOW | 38.56 |  |
| ค้นหา สมชาย | SLOW | 79.87 |  |
| ผู้จัดการ ขอนแก่น | SLOW | 54.81 |  |
| รายชื่อคนในงาน smc | ERROR | 0.00 | name 'query' is not defined |
| นายสมศักดิ์ | SLOW | 39.84 |  |
| ผส.พส. คือใคร | SLOW | 35.13 |  |
| คุณวิชัย อยู่ที่ไหน | SLOW | 77.27 |  |
| คนชื่อ สมพร | SLOW | 42.86 |  |
| หัวหน้างาน noc | SLOW | 61.67 |  |

## Category: TEAM
| Query | Status | Latency (s) | Notes |
|---|---|---|---|
| งาน smc | ERROR | 0.00 | name 'query' is not defined |
| สมาชิก helpdesk | SLOW | 27.31 |  |
| คนในทีม noc | SLOW | 23.63 |  |
| หน้าที่ งาน csoc | SLOW | 66.27 |  |
| งาน support | SLOW | 44.61 |  |
| ทีม develop | SLOW | 37.35 |  |
| ฝ่ายขาย | SLOW | 49.53 |  |
| งาน network | SLOW | 68.90 |  |
| สมาชิก ทีม ip | SLOW | 179.12 |  |
| คนใน แผนกบัญชี | SLOW | 36.26 |  |

## Category: TYPE A (PROCEDURE)
| Query | Status | Latency (s) | Notes |
|---|---|---|---|
| วิธี reset password email | ERROR | 0.00 | name 'query' is not defined |
| ตั้งค่า onu zte | SLOW | 180.41 |  |
| command huawei | ERROR | 0.00 | name 'query' is not defined |
| คู่มือ vpn | SLOW | 174.77 |  |
| ขั้นตอนเบิกของ | SLOW | 40.36 |  |
| วิธีแก้เน็ตช้า | SLOW | 129.82 |  |
| config router cisco | SLOW | 35.96 |  |
| manual mikrotik | ERROR | 0.00 | name 'query' is not defined |
| ขั้นตอนลาป่วย | SLOW | 50.84 |  |
| วิธีใช้ edoc | ERROR | 0.00 | name 'query' is not defined |

## Category: TYPE B (TUTORIAL)
| Query | Status | Latency (s) | Notes |
|---|---|---|---|
| ทำความรู้จัก EtherChannel | ERROR | 0.00 | name 'line_lower' is not defined |
| concept ospf | WARN | 1602.72 | Low Content |
| อธิบาย vlan | WARN | 67.43 | Low Content |
| หลักการ bgp | ERROR | 0.00 | name 'line_lower' is not defined |
| what is mpls | SLOW | 1401.90 |  |
| concept sd-wan | SLOW | 990.34 |  |
| อธิบาย firewall | ERROR | 0.00 | name 'line_lower' is not defined |
| หลักการทำงาน dhcp | ERROR | 0.00 | name 'line_lower' is not defined |
| nat คืออะไร | SLOW | 44.29 |  |
| overview 5g | WARN | 44.51 | Low Content |

## Summary
- **CONTACT**: 10/10 PASS
- **PERSON**: 9/10 PASS
- **TEAM**: 9/10 PASS
- **TYPE A (PROCEDURE)**: 6/10 PASS
- **TYPE B (TUTORIAL)**: 3/10 PASS

**Total: 37/50 (74.0%) | Avg Latency: 163.08s**