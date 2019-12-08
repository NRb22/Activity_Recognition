<?php
session_start();
if($_SESSION['userid']!=null){
  session_destroy();
}
echo"<script> alert('logout');</script>";
echo "<script>location.href='index.php';</script>";
?>
