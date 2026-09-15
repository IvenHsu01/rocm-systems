/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/ras.cc. The suite executes every source line
// and function; branch-heavy network state machines retain intentionally
// unexercised combinations that require integration coverage. External network,
// peer, collective, client, socket-pair, and event-loop operations are TU-local
// seams because this is their sole test consumer.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <poll.h>
#include <vector>

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/signature-drift.h"
#include "os_socket_pair.h"
#include "ras/ras_internal.h"
#include "socket.h"

int RasTestPoll(struct pollfd*, nfds_t, int);
int RasTestClose(int);
int RasTestAtexit(void (*)(void));

#define poll RasTestPoll
#define close RasTestClose
#define atexit RasTestAtexit

namespace {

ncclResult_t DefaultSocketProgress(int, struct ncclSocket*, void*, int size, int* offset, int* closed) {
  *offset = size;
  if (closed) *closed = 0;
  return ncclSuccess;
}

std::function<ncclResult_t(int, struct ncclSocket*, void*, int, int*, int*)> g_socketProgress =
    DefaultSocketProgress;

}  // namespace

ASSERT_HOOK_MATCHES_PROD(g_socketProgress, ncclSocketProgress);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed) {
  return g_socketProgress(op, sock, ptr, size, offset, closed);
}

const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) {
  buf[0] = '\0';
  return buf;
}

#include RAS_CC_PATH

#undef atexit
#undef close
#undef poll

namespace {

extern ncclResult_t g_socketInitResult;
extern ncclResult_t g_socketListenResult;
extern ncclResult_t g_socketPairCreateResult;
extern ncclResult_t g_pairWriteResult;
extern size_t g_pairChunk;
extern std::vector<char> g_pairBytes;
extern size_t g_pairReadPos;
extern int g_socketPairCloseCalls;
extern int g_socketCloseCalls;
extern int g_closeCalls;
extern int g_atexitCalls;
extern int g_pollCalls;
extern std::function<int(struct pollfd*, nfds_t, int)> g_poll;
extern int g_localAddRanksCalls;
extern rasRankInit* g_localAddRanksLast;
extern int g_localAddRanksLastCount;
extern int g_cleanupCalls[4];
void ResetWholeFileSeams();

class RasMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetWholeFileSeams();
    g_socketProgress = DefaultSocketProgress;
    rasInitialized = false;
    rasInitRefCount = 0;
    rasNotificationPipe[0] = rasNotificationPipe[1] = NCCL_SOCKET_PAIR_INVALID;
    std::free(ncclComms);
    ncclComms = nullptr;
    nNcclComms = 0;
    ncclCommsSorted = false;
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
    std::memset(&rasNetListeningSocket, 0, sizeof(rasNetListeningSocket));
  }

  void TearDown() override {
    std::free(rasPfds);
    rasPfds = nullptr;
    nRasPfds = 0;
    std::free(ncclComms);
    ncclComms = nullptr;
    nNcclComms = 0;
    rasInitialized = false;
    rasInitRefCount = 0;
    g_socketProgress = DefaultSocketProgress;
    ResetWholeFileSeams();
  }
};

}  // namespace

TEST_F(RasMicrotest, CommInitAlreadyInitializedRegistersCommAndCopiesListeningAddress) {
  rasInitialized = true;
  rasNetListeningSocket.addr.sin.sin_family = AF_INET;
  rasNetListeningSocket.addr.sin.sin_port = htons(4321);
  ncclComm comm{};
  rasRankInit rank{};
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(&comm, &rank));
  EXPECT_EQ(1, rasInitRefCount);
  EXPECT_EQ(RAS_INCREMENT * 8, nNcclComms);
  EXPECT_EQ(&comm, ncclComms[0]);
  EXPECT_FALSE(ncclCommsSorted);
  EXPECT_EQ(htons(4321), rank.addr.sin.sin_port);
}

TEST_F(RasMicrotest, CommInitReusesVacantCommSlot) {
  rasInitialized = true;
  nNcclComms = 2;
  ncclComms = static_cast<ncclComm**>(std::calloc(2, sizeof(*ncclComms)));
  ncclComm incumbent{};
  ncclComm newcomer{};
  ncclComms[0] = &incumbent;
  rasRankInit rank{};
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(&newcomer, &rank));
  EXPECT_EQ(&newcomer, ncclComms[1]);
  EXPECT_EQ(2, nNcclComms);
}

TEST_F(RasMicrotest, CommInitSocketInitFailureRunsCleanup) {
  g_socketInitResult = ncclSystemError;
  ncclComm comm{};
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(&comm, &rank));
  EXPECT_EQ(1, g_socketPairCloseCalls);
  EXPECT_EQ(1, g_closeCalls);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_FALSE(rasInitialized);
}

TEST_F(RasMicrotest, CommInitListenFailureRunsCleanup) {
  g_socketListenResult = ncclSystemError;
  ncclComm comm{};
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET6;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(&comm, &rank));
  EXPECT_EQ(1, g_socketPairCloseCalls);
  EXPECT_EQ(1, g_closeCalls);
  EXPECT_EQ(1, g_socketCloseCalls);
}

TEST_F(RasMicrotest, CommInitSocketPairFailureRunsCleanup) {
  g_socketPairCreateResult = ncclSystemError;
  ncclComm comm{};
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET;
  EXPECT_EQ(ncclSystemError, ncclRasCommInit(&comm, &rank));
  EXPECT_EQ(1, g_socketPairCloseCalls);
  EXPECT_EQ(1, g_closeCalls);
  EXPECT_EQ(1, g_socketCloseCalls);
  EXPECT_EQ(0, g_atexitCalls);
}

TEST_F(RasMicrotest, CommInitColdSuccessStartsThreadAndPublishesAddress) {
  rasNotification terminate{};
  terminate.type = RAS_TERMINATE;
  const char* bytes = reinterpret_cast<const char*>(&terminate);
  g_pairBytes.assign(bytes, bytes + sizeof(terminate));
  g_poll = [](pollfd* fds, nfds_t, int) {
    while (!rasInitialized) std::this_thread::yield();
    fds[0].revents = POLLIN;
    return 1;
  };

  ncclComm comm{};
  rasRankInit rank{};
  rank.addr.sin.sin_family = AF_INET6;
  ASSERT_EQ(ncclSuccess, ncclRasCommInit(&comm, &rank));
  ASSERT_TRUE(rasThread.joinable());
  rasThread.join();
  EXPECT_EQ(1, g_atexitCalls);
  EXPECT_EQ(1, g_pollCalls);
  EXPECT_EQ(0, rasInitRefCount);
  EXPECT_FALSE(rasInitialized);
}

TEST_F(RasMicrotest, CommFiniUninitializedIsNoOp) {
  ncclComm comm{};
  EXPECT_EQ(ncclSuccess, ncclRasCommFini(&comm));
  EXPECT_EQ(0, rasInitRefCount);
}

TEST_F(RasMicrotest, CommFiniRemovesMatchingCommAndDropsReference) {
  rasInitialized = true;
  rasInitRefCount = 1;
  nNcclComms = 2;
  ncclComms = static_cast<ncclComm**>(std::calloc(2, sizeof(*ncclComms)));
  ncclComm first{};
  ncclComm second{};
  ncclComms[0] = &first;
  ncclComms[1] = &second;
  ncclCommsSorted = true;
  EXPECT_EQ(ncclSuccess, ncclRasCommFini(&second));
  EXPECT_EQ(&first, ncclComms[0]);
  EXPECT_EQ(nullptr, ncclComms[1]);
  EXPECT_FALSE(ncclCommsSorted);
  EXPECT_EQ(0, rasInitRefCount);
}

TEST_F(RasMicrotest, AddRanksBeforeInitializationIsIgnored) {
  rasRankInit ranks[2]{};
  EXPECT_EQ(ncclSuccess, ncclRasAddRanks(ranks, 2));
  EXPECT_TRUE(g_pairBytes.empty());
}

TEST_F(RasMicrotest, AddRanksWritesCompleteNotificationAcrossPartialWrites) {
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  g_pairChunk = 3;
  rasRankInit ranks[2]{};
  ASSERT_EQ(ncclSuccess, ncclRasAddRanks(ranks, 2));
  ASSERT_EQ(sizeof(rasNotification), g_pairBytes.size());
  const auto* msg = reinterpret_cast<const rasNotification*>(g_pairBytes.data());
  EXPECT_EQ(RAS_ADD_RANKS, msg->type);
  EXPECT_EQ(ranks, msg->addRanks.ranks);
  EXPECT_EQ(2, msg->addRanks.nranks);
}

TEST_F(RasMicrotest, AddRanksPropagatesSocketPairWriteFailure) {
  rasInitialized = true;
  g_pairWriteResult = ncclSystemError;
  rasRankInit rank{};
  EXPECT_EQ(ncclSystemError, ncclRasAddRanks(&rank, 1));
}

TEST_F(RasMicrotest, LocalHandleDispatchesAddRanksAndIgnoresHandlerFailure) {
  rasNotification msg{};
  rasRankInit ranks[2]{};
  msg.type = RAS_ADD_RANKS;
  msg.addRanks.ranks = ranks;
  msg.addRanks.nranks = 2;
  const char* bytes = reinterpret_cast<const char*>(&msg);
  g_pairBytes.assign(bytes, bytes + sizeof(msg));
  g_pairChunk = 2;
  bool terminate = false;
  EXPECT_EQ(ncclSuccess, rasLocalHandle(&terminate));
  EXPECT_EQ(1, g_localAddRanksCalls);
  EXPECT_EQ(ranks, g_localAddRanksLast);
  EXPECT_EQ(2, g_localAddRanksLastCount);
  EXPECT_FALSE(terminate);
}

TEST_F(RasMicrotest, LocalHandleTerminateSetsFlag) {
  rasNotification msg{};
  msg.type = RAS_TERMINATE;
  const char* bytes = reinterpret_cast<const char*>(&msg);
  g_pairBytes.assign(bytes, bytes + sizeof(msg));
  bool terminate = false;
  EXPECT_EQ(ncclSuccess, rasLocalHandle(&terminate));
  EXPECT_TRUE(terminate);
}

TEST_F(RasMicrotest, LocalHandleEofAndUnknownTypeReturnErrors) {
  bool terminate = false;
  EXPECT_EQ(ncclSystemError, rasLocalHandle(&terminate));

  rasNotification msg{};
  msg.type = static_cast<rasNotificationType>(99);
  const char* bytes = reinterpret_cast<const char*>(&msg);
  g_pairBytes.assign(bytes, bytes + sizeof(msg));
  g_pairReadPos = 0;
  EXPECT_EQ(ncclInternalError, rasLocalHandle(&terminate));
}

TEST_F(RasMicrotest, ThreadCleanupResetsAllGlobalState) {
  rasInitialized = true;
  rasInitRefCount = 3;
  nNcclComms = 1;
  ncclComms = static_cast<ncclComm**>(std::calloc(1, sizeof(*ncclComms)));
  nRasPfds = 1;
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  rasThreadCleanup();
  EXPECT_EQ(1, g_cleanupCalls[0]);
  EXPECT_EQ(1, g_cleanupCalls[1]);
  EXPECT_EQ(1, g_cleanupCalls[2]);
  EXPECT_EQ(1, g_cleanupCalls[3]);
  EXPECT_FALSE(rasInitialized);
  EXPECT_EQ(0, rasInitRefCount);
  EXPECT_EQ(nullptr, ncclComms);
  EXPECT_EQ(0, nNcclComms);
  EXPECT_EQ(nullptr, rasPfds);
  EXPECT_EQ(0, nRasPfds);
}

namespace {

ncclResult_t g_socketInitResult = ncclSuccess;
ncclResult_t g_socketListenResult = ncclSuccess;
ncclResult_t g_socketPairCreateResult = ncclSuccess;
ncclResult_t g_socketGetFdResult = ncclSuccess;
int g_socketGetFdValue = 41;
int g_socketCloseCalls = 0;
int g_socketPairCloseCalls = 0;
int g_closeCalls = 0;
int g_atexitCalls = 0;
int g_pollCalls = 0;
std::function<int(struct pollfd*, nfds_t, int)> g_poll = [](struct pollfd*, nfds_t, int) { return 0; };
std::vector<char> g_pairBytes;
size_t g_pairReadPos = 0;
size_t g_pairChunk = SIZE_MAX;
ncclResult_t g_pairWriteResult = ncclSuccess;
ncclResult_t g_pairReadResult = ncclSuccess;

int g_localAddRanksCalls = 0;
rasRankInit* g_localAddRanksLast = nullptr;
int g_localAddRanksLastCount = 0;
int g_cleanupCalls[4] = {};

int g_keepAliveCalls = 0;
int g_peersUpdateCalls = 0;
int g_collReqCalls = 0;
int g_collRespCalls = 0;
ncclResult_t g_dispatchResult = ncclSuccess;

rasConnection* g_connFindResult = nullptr;
rasConnection g_newConn{};
ncclResult_t g_newConnResult = ncclSuccess;
int g_socketTerminateCalls = 0;
rasSocket* g_lastTerminatedSocket = nullptr;
int g_socketCompareResult = 0;
int g_clientsNotifyCalls = 0;
int g_peerFindResult = -1;
int g_linkUpdateCalls = 0;
int g_sendPeersUpdateCalls = 0;
bool g_peerDead = false;
int g_connDisconnectCalls = 0;
int g_peerDeclareDeadCalls = 0;
int g_netAcceptCalls = 0;
int g_clientAcceptCalls = 0;
int g_sockEventCalls = 0;
int g_clientEventCalls = 0;
int g_timeoutCalls[4] = {};
int g_lastSockPollIdx = -1;
int g_lastClientPollIdx = -1;
int64_t g_nextWakeupOverride = 0;

void ResetWholeFileSeams() {
  g_socketInitResult = ncclSuccess;
  g_socketListenResult = ncclSuccess;
  g_socketPairCreateResult = ncclSuccess;
  g_socketGetFdResult = ncclSuccess;
  g_socketGetFdValue = 41;
  g_socketCloseCalls = g_socketPairCloseCalls = g_closeCalls = g_atexitCalls = g_pollCalls = 0;
  g_poll = [](struct pollfd*, nfds_t, int) { return 0; };
  g_pairBytes.clear();
  g_pairReadPos = 0;
  g_pairChunk = SIZE_MAX;
  g_pairWriteResult = g_pairReadResult = ncclSuccess;
  g_localAddRanksCalls = 0;
  g_localAddRanksLast = nullptr;
  g_localAddRanksLastCount = 0;
  std::memset(g_cleanupCalls, 0, sizeof(g_cleanupCalls));
  g_keepAliveCalls = g_peersUpdateCalls = g_collReqCalls = g_collRespCalls = 0;
  g_dispatchResult = ncclSuccess;
  g_connFindResult = nullptr;
  std::memset(&g_newConn, 0, sizeof(g_newConn));
  g_newConnResult = ncclSuccess;
  g_socketTerminateCalls = 0;
  g_lastTerminatedSocket = nullptr;
  g_socketCompareResult = 0;
  g_clientsNotifyCalls = 0;
  g_peerFindResult = -1;
  g_linkUpdateCalls = g_sendPeersUpdateCalls = 0;
  g_peerDead = false;
  g_connDisconnectCalls = g_peerDeclareDeadCalls = 0;
  g_netAcceptCalls = g_clientAcceptCalls = 0;
  g_sockEventCalls = g_clientEventCalls = 0;
  std::memset(g_timeoutCalls, 0, sizeof(g_timeoutCalls));
  g_lastSockPollIdx = g_lastClientPollIdx = -1;
  g_nextWakeupOverride = 0;
  rasSocketsHead = nullptr;
  rasClientsHead = nullptr;
}

}  // namespace

int rasClientListeningSocket = 43;
rasSocket* rasSocketsHead = nullptr;
rasClient* rasClientsHead = nullptr;
rasLink rasNextLink{};
rasLink rasPrevLink{};
rasPeerInfo* rasPeers = nullptr;
int nRasPeers = 0;
uint64_t rasPeersHash = 0;
union ncclSocketAddress* rasDeadPeers = nullptr;
int nRasDeadPeers = 0;
uint64_t rasDeadPeersHash = 0;

int RasTestPoll(struct pollfd* fds, nfds_t n, int timeout) {
  ++g_pollCalls;
  return g_poll(fds, n, timeout);
}
int RasTestClose(int) {
  ++g_closeCalls;
  return 0;
}
int RasTestAtexit(void (*)(void)) {
  ++g_atexitCalls;
  return 0;
}

uint64_t ncclSocketDefaultMagic() { return 0x1234; }
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t,
                            enum ncclSocketType, volatile uint32_t*, int, int) {
  if (g_socketInitResult == ncclSuccess && sock && addr) std::memcpy(&sock->addr, addr, sizeof(*addr));
  return g_socketInitResult;
}
ncclResult_t ncclSocketListen(struct ncclSocket*) { return g_socketListenResult; }
ncclResult_t ncclSocketGetFd(struct ncclSocket*, ncclSocketDescriptor* fd) {
  if (g_socketGetFdResult == ncclSuccess && fd) *fd = g_socketGetFdValue;
  return g_socketGetFdResult;
}
ncclResult_t ncclSocketClose(struct ncclSocket*, bool) {
  ++g_socketCloseCalls;
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]) {
  if (g_socketPairCreateResult == ncclSuccess) {
    pair[0] = 51;
    pair[1] = 52;
  }
  return g_socketPairCreateResult;
}
ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]) {
  ++g_socketPairCloseCalls;
  pair[0] = pair[1] = NCCL_SOCKET_PAIR_INVALID;
  return ncclSuccess;
}
ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor, const void* buf, size_t len, size_t* written) {
  if (g_pairWriteResult != ncclSuccess) return g_pairWriteResult;
  size_t n = std::min(len, g_pairChunk);
  const char* bytes = static_cast<const char*>(buf);
  g_pairBytes.insert(g_pairBytes.end(), bytes, bytes + n);
  *written = n;
  return ncclSuccess;
}
ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor, void* buf, size_t len, size_t* nread) {
  if (g_pairReadResult != ncclSuccess) return g_pairReadResult;
  size_t available = g_pairBytes.size() - g_pairReadPos;
  size_t n = std::min(std::min(len, available), g_pairChunk);
  if (n) std::memcpy(buf, g_pairBytes.data() + g_pairReadPos, n);
  g_pairReadPos += n;
  *nread = n;
  return ncclSuccess;
}
void ncclSetThreadName(std::thread&, const char*, ...) {}

ncclResult_t rasClientInitSocket() { return ncclSuccess; }
ncclResult_t rasClientAcceptNewSocket() {
  ++g_clientAcceptCalls;
  return ncclSuccess;
}
void rasClientEventLoop(struct rasClient* client, int pollIdx) {
  ++g_clientEventCalls;
  g_lastClientPollIdx = pollIdx;
}
void rasClientSupportTerminate() { ++g_cleanupCalls[0]; }
void rasNetTerminate() { ++g_cleanupCalls[1]; }
void rasCollectivesTerminate() { ++g_cleanupCalls[2]; }
void rasPeersTerminate() { ++g_cleanupCalls[3]; }
void rasSocksHandleTimeouts(int64_t, int64_t* nextWakeup) {
  ++g_timeoutCalls[0];
  if (g_nextWakeupOverride) *nextWakeup = g_nextWakeupOverride;
}
void rasConnsHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[1]; }
void rasNetHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[2]; }
void rasCollsHandleTimeouts(int64_t, int64_t*) { ++g_timeoutCalls[3]; }
ncclResult_t rasNetAcceptNewSocket() {
  ++g_netAcceptCalls;
  return ncclSuccess;
}
void rasSockEventLoop(struct rasSocket*, int pollIdx) {
  ++g_sockEventCalls;
  g_lastSockPollIdx = pollIdx;
}

ncclResult_t rasLocalHandleAddRanks(struct rasRankInit* ranks, int nranks) {
  ++g_localAddRanksCalls;
  g_localAddRanksLast = ranks;
  g_localAddRanksLastCount = nranks;
  return ncclSuccess;
}
ncclResult_t rasMsgHandleKeepAlive(const struct rasMsg*, struct rasSocket*) {
  ++g_keepAliveCalls;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandlePeersUpdate(struct rasMsg*, struct rasSocket*) {
  ++g_peersUpdateCalls;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandleCollReq(struct rasMsg*, struct rasSocket*) {
  ++g_collReqCalls;
  return g_dispatchResult;
}
ncclResult_t rasMsgHandleCollResp(struct rasMsg*, struct rasSocket*) {
  ++g_collRespCalls;
  return g_dispatchResult;
}

rasConnection* rasConnFind(const union ncclSocketAddress*) { return g_connFindResult; }
ncclResult_t getNewConnEntry(struct rasConnection** conn) {
  if (g_newConnResult == ncclSuccess) *conn = &g_newConn;
  return g_newConnResult;
}
void rasSocketTerminate(struct rasSocket* sock, bool, uint64_t, bool) {
  ++g_socketTerminateCalls;
  g_lastTerminatedSocket = sock;
}
int ncclSocketsCompare(const void*, const void*) { return g_socketCompareResult; }
void rasClientsNotifyEvent(rasEventGroup, const struct rasEventNotification*) { ++g_clientsNotifyCalls; }
int rasPeerFind(const union ncclSocketAddress*) { return g_peerFindResult; }
ncclResult_t rasLinkConnUpdate(struct rasLink*, struct rasConnection*, int) {
  ++g_linkUpdateCalls;
  return ncclSuccess;
}
ncclResult_t rasConnSendPeersUpdate(struct rasConnection*, const struct rasPeerInfo*, int) {
  ++g_sendPeersUpdateCalls;
  return ncclSuccess;
}
bool rasPeerIsDead(const union ncclSocketAddress*) { return g_peerDead; }
void rasConnDisconnect(const union ncclSocketAddress*) { ++g_connDisconnectCalls; }
ncclResult_t rasPeerDeclareDead(const union ncclSocketAddress*) {
  ++g_peerDeclareDeadCalls;
  return ncclSuccess;
}

namespace {

struct OwnedMsg {
  rasMsg* ptr = nullptr;
  explicit OwnedMsg(size_t len) { EXPECT_EQ(ncclSuccess, rasMsgAlloc(&ptr, len)); }
  ~OwnedMsg() { rasMsgFree(ptr); }
  rasMsg* release() {
    rasMsg* out = ptr;
    ptr = nullptr;
    return out;
  }
};

}  // namespace

TEST_F(RasMicrotest, MessageLengthsCoverEveryFixedAndCollectiveType) {
  EXPECT_EQ(offsetof(rasMsg, connInit) + sizeof(rasMsg{}.connInit), rasMsgLength(RAS_MSG_CONNINIT));
  EXPECT_EQ(offsetof(rasMsg, connInitAck) + sizeof(rasMsg{}.connInitAck), rasMsgLength(RAS_MSG_CONNINITACK));
  EXPECT_EQ(offsetof(rasMsg, keepAlive) + sizeof(rasMsg{}.keepAlive), rasMsgLength(RAS_MSG_KEEPALIVE));
  EXPECT_EQ(offsetof(rasMsg, peersUpdate) + sizeof(rasMsg{}.peersUpdate), rasMsgLength(RAS_MSG_PEERSUPDATE));
  EXPECT_EQ(offsetof(rasMsg, collResp) + sizeof(rasMsg{}.collResp), rasMsgLength(RAS_MSG_COLLRESP));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_BC_DEADPEER),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_BC_DEADPEER));
  EXPECT_EQ(offsetof(rasCollRequest, conns) + sizeof(rasCollRequest{}.conns), rasCollDataLength(RAS_COLL_CONNS));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_COLL_CONNS),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_CONNS));
  EXPECT_EQ(offsetof(rasCollRequest, comms) + sizeof(rasCollRequest{}.comms), rasCollDataLength(RAS_COLL_COMMS));
  EXPECT_EQ(offsetof(rasMsg, collReq) + rasCollDataLength(RAS_COLL_COMMS),
            rasMsgLength(RAS_MSG_COLLREQ, RAS_COLL_COMMS));
  EXPECT_EQ(0u, rasCollDataLength(RAS_MSG_NONE));
  EXPECT_EQ(0u, rasCollDataLength(static_cast<rasCollectiveType>(-1)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(0)));
  EXPECT_EQ(0u, rasMsgLength(static_cast<rasMsgType>(-1)));
}

TEST_F(RasMicrotest, TerminateUninitializedIsNoOp) {
  rasTerminate();
  EXPECT_TRUE(g_pairBytes.empty());
}

TEST_F(RasMicrotest, TerminateNotifiesAndJoinsThread) {
  rasInitialized = true;
  rasNotificationPipe[1] = 52;
  rasThread = std::thread([] {});
  rasTerminate();
  EXPECT_FALSE(rasThread.joinable());
  ASSERT_EQ(sizeof(rasNotification), g_pairBytes.size());
  EXPECT_EQ(RAS_TERMINATE, reinterpret_cast<const rasNotification*>(g_pairBytes.data())->type);
}

TEST_F(RasMicrotest, MessageDispatchRoutesExternalTypesAndPropagatesErrors) {
  rasSocket sock{};
  rasMsg msg{};
  for (auto entry : {std::pair{RAS_MSG_KEEPALIVE, &g_keepAliveCalls},
                     std::pair{RAS_MSG_PEERSUPDATE, &g_peersUpdateCalls},
                     std::pair{RAS_MSG_COLLREQ, &g_collReqCalls},
                     std::pair{RAS_MSG_COLLRESP, &g_collRespCalls}}) {
    msg.type = entry.first;
    EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
    EXPECT_EQ(1, *entry.second);
  }
  g_dispatchResult = ncclSystemError;
  msg.type = RAS_MSG_KEEPALIVE;
  EXPECT_EQ(ncclSystemError, rasMsgHandle(&msg, &sock));
  msg.type = static_cast<rasMsgType>(99);
  EXPECT_EQ(ncclInternalError, rasMsgHandle(&msg, &sock));
}

TEST_F(RasMicrotest, ConnInitVersionMismatchSendsNackAndTerminatesSocket) {
  rasSocket sock{};
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE - 1;
  int progressCalls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++progressCalls;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclInvalidUsage, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, progressCalls);
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&sock, g_lastTerminatedSocket);
}

TEST_F(RasMicrotest, ConnInitKnownDeadPeerNacksWithoutCreatingConnection) {
  g_peerDead = true;
  rasSocket sock{};
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(nullptr, sock.conn);
}

TEST_F(RasMicrotest, ConnInitCreatesReadyConnectionAndQueuesAck) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasSocket sock{};
  sock.pfd = 0;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  msg.connInit.peersHash = rasPeersHash;
  msg.connInit.deadPeersHash = rasDeadPeersHash;
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(RAS_SOCK_READY, sock.status);
  EXPECT_EQ(&g_newConn, sock.conn);
  EXPECT_EQ(&sock, g_newConn.sock);
  EXPECT_EQ(1, g_clientsNotifyCalls);
  rasMsgMeta* meta = ncclIntruQueueHead(&g_newConn.sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(RAS_MSG_CONNINITACK, meta->msg.type);
  EXPECT_EQ(0, meta->msg.connInitAck.nack);
  rasMsgFree(&ncclIntruQueueDequeue(&g_newConn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnInitHashMismatchSendsPeersUpdateAndUpdatesBothLinks) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  g_peerFindResult = 3;
  rasPeersHash = 10;
  rasDeadPeersHash = 20;
  rasSocket sock{};
  sock.pfd = 0;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  msg.connInit.peersHash = 11;
  msg.connInit.deadPeersHash = 21;
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(2, g_linkUpdateCalls);
  EXPECT_EQ(1, g_sendPeersUpdateCalls);
  EXPECT_EQ(11u, g_newConn.lastRecvPeersHash);
  EXPECT_EQ(21u, g_newConn.lastRecvDeadPeersHash);
  rasMsgFree(&ncclIntruQueueDequeue(&g_newConn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnInitExistingLowerAddressRejectsNewSocket) {
  rasConnection existing{};
  rasSocket existingSock{};
  existing.sock = &existingSock;
  g_connFindResult = &existing;
  g_socketCompareResult = -1;
  rasSocket incoming{};
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&incoming, g_lastTerminatedSocket);
  EXPECT_EQ(nullptr, incoming.conn);
}

TEST_F(RasMicrotest, ConnInitExistingHigherAddressReplacesOldSocket) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasConnection existing{};
  rasSocket oldSock{};
  existing.sock = &oldSock;
  g_connFindResult = &existing;
  g_socketCompareResult = 1;
  rasSocket incoming{};
  incoming.pfd = 0;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINIT;
  msg.connInit.ncclVersion = NCCL_VERSION_CODE;
  ASSERT_EQ(ncclSuccess, rasMsgHandle(&msg, &incoming));
  EXPECT_EQ(1, g_socketTerminateCalls);
  EXPECT_EQ(&oldSock, g_lastTerminatedSocket);
  EXPECT_EQ(&incoming, existing.sock);
  EXPECT_EQ(&existing, incoming.conn);
  rasMsgFree(&ncclIntruQueueDequeue(&existing.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnInitAckNackDeclaresPeerDead) {
  rasSocket sock{};
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 1;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);
}

TEST_F(RasMicrotest, ConnInitAckSuccessMarksSocketReady) {
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  rasMsg msg{};
  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 0;
  EXPECT_EQ(ncclSuccess, rasMsgHandle(&msg, &sock));
  EXPECT_EQ(RAS_SOCK_READY, sock.status);
}

TEST_F(RasMicrotest, DeadPeerBroadcastHandlesNewAndKnownPeers) {
  rasCollRequest req{};
  rasCollRequest* reqPtr = &req;
  size_t reqLen = 0;
  bool done = true;
  rasMsgHandleBCDeadPeer(&reqPtr, &reqLen, &done);
  EXPECT_EQ(rasCollDataLength(RAS_BC_DEADPEER), reqLen);
  EXPECT_FALSE(done);
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);

  g_peerDead = true;
  done = false;
  rasMsgHandleBCDeadPeer(&reqPtr, &reqLen, &done);
  EXPECT_TRUE(done);
  EXPECT_EQ(1, g_connDisconnectCalls);
  EXPECT_EQ(1, g_peerDeclareDeadCalls);
}

TEST_F(RasMicrotest, NetSendNackStopsAfterPartialLengthAndPropagatesError) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 1;
    *closed = 0;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclSuccess, rasNetSendNack(&sock));

  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  EXPECT_EQ(ncclSystemError, rasNetSendNack(&sock));
}

TEST_F(RasMicrotest, ThreadMainSocketFdFailureRunsCleanup) {
  rasNotificationPipe[0] = 51;
  g_socketGetFdResult = ncclSystemError;
  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(1, g_cleanupCalls[0]);
  EXPECT_EQ(1, g_cleanupCalls[1]);
  EXPECT_EQ(1, g_cleanupCalls[2]);
  EXPECT_EQ(1, g_cleanupCalls[3]);
  EXPECT_EQ(nullptr, rasPfds);
}

TEST_F(RasMicrotest, ThreadMainDispatchesEveryFdClassAndTerminatesCleanly) {
  rasNotificationPipe[0] = 51;
  rasClientListeningSocket = 43;
  rasPfds = static_cast<pollfd*>(std::calloc(8, sizeof(*rasPfds)));
  nRasPfds = 8;
  for (int i = 0; i < nRasPfds; ++i) rasPfds[i].fd = NCCL_INVALID_SOCKET;
  rasPfds[3].fd = 61;
  rasPfds[4].fd = 62;

  rasSocket rasSock{};
  rasSock.sock.socketDescriptor = 61;
  rasSocketsHead = &rasSock;
  rasClient skippedClient{};
  skippedClient.sock = 63;
  rasClient client{};
  client.sock = 62;
  skippedClient.next = &client;
  rasClientsHead = &skippedClient;

  rasNotification terminate{};
  terminate.type = RAS_TERMINATE;
  const char* bytes = reinterpret_cast<const char*>(&terminate);
  g_pairBytes.assign(bytes, bytes + sizeof(terminate));

  std::vector<int> timeouts;
  g_poll = [&](pollfd* fds, nfds_t n, int timeout) {
    timeouts.push_back(timeout);
    EXPECT_EQ(8u, n);
    if (g_pollCalls == 1) {
      fds[1].revents = POLLIN;
      fds[2].revents = POLLIN;
      fds[3].revents = POLLIN;
      fds[4].revents = POLLIN;
      return 4;
    }
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(2, g_pollCalls);
  ASSERT_EQ(2u, timeouts.size());
  EXPECT_EQ(1000, timeouts[0]);
  EXPECT_GT(timeouts[1], 0);
  EXPECT_EQ(1, g_netAcceptCalls);
  EXPECT_EQ(1, g_clientAcceptCalls);
  EXPECT_EQ(1, g_sockEventCalls);
  EXPECT_EQ(1, g_clientEventCalls);
  EXPECT_EQ(3, g_lastSockPollIdx);
  EXPECT_EQ(4, g_lastClientPollIdx);
  for (int calls : g_timeoutCalls) EXPECT_EQ(1, calls);
  for (int calls : g_cleanupCalls) EXPECT_EQ(1, calls);
}

TEST_F(RasMicrotest, ThreadMainToleratesPollErrorAndInvalidFdBeforeTerminating) {
  rasNotificationPipe[0] = 51;
  rasPfds = static_cast<pollfd*>(std::calloc(4, sizeof(*rasPfds)));
  nRasPfds = 4;
  for (int i = 0; i < nRasPfds; ++i) rasPfds[i].fd = NCCL_INVALID_SOCKET;
  rasPfds[3].fd = 77;

  rasNotification terminate{};
  terminate.type = RAS_TERMINATE;
  const char* bytes = reinterpret_cast<const char*>(&terminate);
  g_pairBytes.assign(bytes, bytes + sizeof(terminate));
  g_nextWakeupOverride = 1;

  std::vector<int> timeouts;
  g_poll = [&](pollfd* fds, nfds_t, int timeout) {
    timeouts.push_back(timeout);
    if (g_pollCalls == 1) {
      errno = EBADF;
      return -1;
    }
    if (g_pollCalls == 2) {
      fds[3].revents = POLLNVAL;
      return 1;
    }
    EXPECT_EQ(POLL_FD_IGNORE, fds[3].fd);
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(3, g_pollCalls);
  ASSERT_EQ(3u, timeouts.size());
  EXPECT_EQ(1000, timeouts[0]);
  EXPECT_EQ(1, timeouts[1]);
  EXPECT_EQ(1, timeouts[2]);
  for (int calls : g_timeoutCalls) EXPECT_EQ(2, calls);
}

TEST_F(RasMicrotest, ThreadMainTreatsInterruptedPollAsAnEmptyIteration) {
  rasNotificationPipe[0] = 51;
  rasNotification terminate{};
  terminate.type = RAS_TERMINATE;
  const char* bytes = reinterpret_cast<const char*>(&terminate);
  g_pairBytes.assign(bytes, bytes + sizeof(terminate));
  g_poll = [](pollfd* fds, nfds_t, int) {
    if (g_pollCalls == 1) {
      errno = EINTR;
      return -1;
    }
    fds[0].revents = POLLIN;
    return 1;
  };

  EXPECT_EQ(nullptr, rasThreadMain(nullptr));
  EXPECT_EQ(2, g_pollCalls);
  for (int calls : g_timeoutCalls) EXPECT_EQ(1, calls);
}

TEST_F(RasMicrotest, MsgAllocReturnsZeroedPayloadAndFreeAcceptsNull) {
  rasMsg* msg = nullptr;
  ASSERT_EQ(ncclSuccess, rasMsgAlloc(&msg, rasMsgLength(RAS_MSG_KEEPALIVE)));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_NONE, msg->type);
  rasMsgFree(msg);
  rasMsgFree(nullptr);
}

TEST_F(RasMicrotest, ConnEnqueueBackInitializesMetadataAndArmsReadySocket) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasSocket sock{};
  sock.status = RAS_SOCK_READY;
  sock.pfd = 0;
  rasConnection conn{};
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE), false);

  rasMsgMeta* meta = ncclIntruQueueHead(&conn.sendQ);
  ASSERT_NE(nullptr, meta);
  EXPECT_EQ(0, meta->offset);
  EXPECT_EQ((int)rasMsgLength(RAS_MSG_KEEPALIVE), meta->length);
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnEnqueueFrontPrecedesExistingMessage) {
  rasConnection conn{};
  OwnedMsg first(rasMsgLength(RAS_MSG_KEEPALIVE));
  OwnedMsg second(rasMsgLength(RAS_MSG_CONNINIT));
  first.ptr->type = RAS_MSG_KEEPALIVE;
  second.ptr->type = RAS_MSG_CONNINIT;
  rasMsg* firstRaw = first.release();
  rasMsg* secondRaw = second.release();
  rasConnEnqueueMsg(&conn, firstRaw, rasMsgLength(RAS_MSG_KEEPALIVE), false);
  rasConnEnqueueMsg(&conn, secondRaw, rasMsgLength(RAS_MSG_CONNINIT), true);
  EXPECT_EQ(secondRaw, &ncclIntruQueueHead(&conn.sendQ)->msg);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnEnqueueHandshakeArmsOnlyConnInitMessage) {
  rasPfds = static_cast<pollfd*>(std::calloc(1, sizeof(*rasPfds)));
  nRasPfds = 1;
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  sock.pfd = 0;
  rasConnection conn{};
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_CONNINIT));
  owned.ptr->type = RAS_MSG_CONNINIT;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_CONNINIT));
  EXPECT_NE(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);

  rasPfds[0].events = 0;
  OwnedMsg blocked(rasMsgLength(RAS_MSG_KEEPALIVE));
  blocked.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, blocked.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  EXPECT_EQ(0, rasPfds[0].events & POLLOUT);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendEmptyQueueReportsAllSent) {
  rasConnection conn{};
  rasSocket sock{};
  conn.sock = &sock;
  int closed = -1;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, closed);
  EXPECT_TRUE(allSent);
}

TEST_F(RasMicrotest, ConnSendHandshakeBlocksNonInitMessageWithoutCallingSocket) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_HANDSHAKE;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int, int*, int*) {
    ++calls;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(0, calls);
  EXPECT_TRUE(allSent);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendCompleteMessageDequeuesIt) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED; // enqueue need not arm rasPfds; send itself accepts this state
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void*, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_SEND, op);
    ++calls;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_TRUE(allSent);
  EXPECT_TRUE(ncclIntruQueueEmpty(&conn.sendQ));
}

TEST_F(RasMicrotest, ConnSendPartialLengthKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_FALSE(allSent);
  EXPECT_EQ(2, ncclIntruQueueHead(&conn.sendQ)->offset);
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendClosedSocketReturnsWithoutDequeuing) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  int closed = 0;
  bool allSent = false;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(1, closed);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendPartialBodyKeepsMessageQueued) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void*, int size, int* offset, int* closed) {
    ++calls;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  int closed;
  bool allSent = true;
  EXPECT_EQ(ncclSuccess, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_EQ(2, calls);
  EXPECT_FALSE(allSent);
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, ConnSendSocketErrorPropagates) {
  rasConnection conn{};
  rasSocket sock{};
  sock.status = RAS_SOCK_CLOSED;
  conn.sock = &sock;
  OwnedMsg owned(rasMsgLength(RAS_MSG_KEEPALIVE));
  owned.ptr->type = RAS_MSG_KEEPALIVE;
  rasConnEnqueueMsg(&conn, owned.release(), rasMsgLength(RAS_MSG_KEEPALIVE));
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  int closed;
  bool allSent;
  EXPECT_EQ(ncclSystemError, rasConnSendMsg(&conn, &closed, &allSent));
  EXPECT_FALSE(ncclIntruQueueEmpty(&conn.sendQ));
  rasMsgFree(&ncclIntruQueueDequeue(&conn.sendQ)->msg);
}

TEST_F(RasMicrotest, MsgRecvCompletesLengthThenBodyAndResetsSocketState) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
    } else {
      auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(int));
      msg->type = RAS_MSG_KEEPALIVE;
    }
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(2, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, MsgRecvPartialLengthReturnsWithoutAllocatingBody) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int* offset, int* closed) {
    *offset = 2;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
  EXPECT_EQ(2, sock.recvOffset);
}

TEST_F(RasMicrotest, MsgRecvClosedDuringLengthReturnsImmediately) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int* closed) {
    *closed = 1;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvSocketErrorPropagates) {
  rasSocket sock{};
  g_socketProgress = [](int, ncclSocket*, void*, int, int*, int*) { return ncclSystemError; };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSystemError, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  EXPECT_EQ(nullptr, sock.recvMsg);
}

TEST_F(RasMicrotest, MsgRecvClosedDuringBodyPreservesAllocatedMessage) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) {
      *static_cast<int*>(ptr) = msgLen;
      *offset = size;
      *closed = 0;
    } else {
      *closed = 1;
    }
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed = 0;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(1, closed);
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvPartialBodyPreservesProgressForNextCall) {
  rasSocket sock{};
  const int msgLen = rasMsgLength(RAS_MSG_KEEPALIVE);
  int calls = 0;
  g_socketProgress = [&](int, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    ++calls;
    if (calls == 1) *static_cast<int*>(ptr) = msgLen;
    *offset = calls == 1 ? size : size - 1;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  EXPECT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  EXPECT_EQ(nullptr, msg);
  ASSERT_NE(nullptr, sock.recvMsg);
  EXPECT_EQ(msgLen + (int)sizeof(int) - 1, sock.recvOffset);
  std::free(sock.recvMsg);
  sock.recvMsg = nullptr;
}

TEST_F(RasMicrotest, MsgRecvResumesPartialBodyWithoutRereadingLength) {
  rasSocket sock{};
  sock.recvLength = rasMsgLength(RAS_MSG_KEEPALIVE);
  sock.recvOffset = sizeof(sock.recvLength) + 3;
  sock.recvMsg = static_cast<rasMsg*>(std::calloc(1, sock.recvLength));
  int calls = 0;
  g_socketProgress = [&](int op, ncclSocket*, void* ptr, int size, int* offset, int* closed) {
    EXPECT_EQ(NCCL_SOCKET_RECV, op);
    EXPECT_EQ(sock.recvLength + (int)sizeof(sock.recvLength), size);
    ++calls;
    auto* msg = reinterpret_cast<rasMsg*>(static_cast<char*>(ptr) + sizeof(sock.recvLength));
    msg->type = RAS_MSG_KEEPALIVE;
    *offset = size;
    *closed = 0;
    return ncclSuccess;
  };
  rasMsg* msg = nullptr;
  int closed;
  ASSERT_EQ(ncclSuccess, rasMsgRecv(&sock, &msg, &closed));
  ASSERT_NE(nullptr, msg);
  EXPECT_EQ(RAS_MSG_KEEPALIVE, msg->type);
  EXPECT_EQ(1, calls);
  EXPECT_EQ(0, sock.recvOffset);
  EXPECT_EQ(0, sock.recvLength);
  EXPECT_EQ(nullptr, sock.recvMsg);
  std::free(msg);
}

TEST_F(RasMicrotest, GetNewPollEntryGrowsInChunksAndReusesVacancies) {
  int first = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&first));
  EXPECT_EQ(0, first);
  EXPECT_EQ(RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[first].fd);

  rasPfds[0].fd = 7;
  int second = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&second));
  EXPECT_EQ(1, second);
  rasPfds[0].fd = NCCL_INVALID_SOCKET;
  int reused = -1;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&reused));
  EXPECT_EQ(0, reused);
}

TEST_F(RasMicrotest, GetNewPollEntryExpandsAgainWhenEverySlotIsOccupied) {
  int index;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  ASSERT_EQ(RAS_INCREMENT, nRasPfds);
  for (int i = 0; i < nRasPfds; ++i) rasPfds[i].fd = i + 10;
  ASSERT_EQ(ncclSuccess, rasGetNewPollEntry(&index));
  EXPECT_EQ(RAS_INCREMENT, index);
  EXPECT_EQ(2 * RAS_INCREMENT, nRasPfds);
  EXPECT_EQ(NCCL_INVALID_SOCKET, rasPfds[index].fd);
  EXPECT_EQ(0, rasPfds[index].events);
  EXPECT_EQ(0, rasPfds[index].revents);
}
